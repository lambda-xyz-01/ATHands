import torch
from torch.utils import data

import numpy as np
import os
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import joblib

from config import *


class Text2MotionDataset(data.Dataset):
	"""Dataset for Text2Motion generation task.

	"""
	def __init__(self, opt, split_file):
		self.opt = opt
		self.max_length = 20
		min_motion_len = 24
		self.n_body_joints = opt.num_body_joints
		self.n_joints = opt.num_joints
		
		with cs.open(split_file, 'r') as f:
			id_list = f.read().strip().split('\n')

		data_dict = []
		for name in tqdm(id_list):
			motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
			if (len(motion)) < min_motion_len or (len(motion) >= 200):
				continue

			with cs.open(pjoin(opt.body_text_dir, name + '.txt')) as f:
				body_text = f.read().strip().split('\n')
			with cs.open(pjoin(opt.hand_text_dir, name + '.txt')) as f:
				hand_text = f.read().strip().split('\n')

			if len(hand_text) == 0:
				hand_text = body_text

			data = {'motion': motion,
					'length': len(motion),
					'body_text':body_text,
					'hand_text':hand_text,
									}

			data_dict.append(data)
			
		self.data_dict = data_dict


		meanstd_data_path = pjoin(opt.data_root, "bodyhand_processed_dataset_meanstd.p")

		if(not os.path.exists(meanstd_data_path)):    
			print("Calculating mean std ")
			self.min_max_mean_std_jpos_data = self.extract_min_max_mean_std_from_data()
			joblib.dump(self.min_max_mean_std_jpos_data, meanstd_data_path)
		else:
			print("Loading mean std ")
			self.min_max_mean_std_jpos_data = joblib.load(meanstd_data_path)

		self.global_jpos_min = self.min_max_mean_std_jpos_data['global_jpos_min'].reshape(self.n_joints, 3)[None]
		self.global_jpos_max = self.min_max_mean_std_jpos_data['global_jpos_max'].reshape(self.n_joints, 3)[None]

	def inv_transform(self, data):
		return data * self.std + self.mean

	def real_len(self):
		return len(self.data_dict)

	def extract_min_max_mean_std_from_data(self):
		min_jpos = np.inf*np.ones(self.n_joints*3) 
		max_jpos = -np.inf*np.ones(self.n_joints*3) 
		min_jvel = np.inf*np.ones(self.n_joints*3)
		max_jvel = -np.inf*np.ones(self.n_joints*3) 

		for data in self.data_dict:
			motion = data['motion']
			current_jpos = motion[:, :self.n_joints*3].reshape(-1, self.n_joints*3)

			min_jpos = np.minimum(min_jpos, current_jpos.min(axis=0))
			max_jpos = np.maximum(max_jpos, current_jpos.max(axis=0))

		stats_dict = {}
		stats_dict['global_jpos_min'] = min_jpos 
		stats_dict['global_jpos_max'] = max_jpos 

		return stats_dict

	def normalize_jpos_min_max(self, ori_jpos):
		
		# ori_jpos: T X 22 X 3 
		min_val = self.global_jpos_min
		max_val = self.global_jpos_max
		
		diff = max_val - min_val
		diff[diff == 0] = 1e-8  # set very small value to prevent division by zero
		
		normalized_jpos = (ori_jpos - min_val) / diff
		normalized_jpos = normalized_jpos * 2 - 1 # [-1, 1] range 
		return normalized_jpos # T X 22 X 3

	def de_normalize_jpos_min_max(self, normalized_jpos):
		normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

		min_val = self.global_jpos_min
		max_val = self.global_jpos_max

		diff = max_val - min_val
		diff[diff == 0] = 1e-8  # set very small value to prevent division by zero

		de_jpos = normalized_jpos * diff + min_val
		return de_jpos # T X 22 X 3

	def __len__(self):
		return self.real_len() * self.times


	def __getitem__(self, item):
		idx = item % self.real_len()
		data = self.data_dict[idx]
		motion, m_length = data['motion'], data['length']
		hand_text_list = data['hand_text'] if len(data['hand_text']) > 0 else data['body_text']
	
		hand_text = random.choice(hand_text_list)

		## global pose
		globla_jpos = motion[:, :self.n_joints*3]
		global_rot6d = motion[:, self.n_joints*3:]
		normalized_jpos = self.normalize_jpos_min_max(globla_jpos.reshape(-1, self.n_joints, 3))
		new_motion = np.concatenate((normalized_jpos.reshape(-1, self.n_joints*3), global_rot6d), axis=1) #T*(52*3+52*6)
		body_motion = np.concatenate((new_motion[:, :self.n_body_joints*3],
										new_motion[:, self.n_joints*3:self.n_joints*3+self.n_body_joints*6],
										),
										axis=-1)
		## local hand pose
		global_rot6d = global_rot6d.reshape(-1, self.n_joints, 6)
		left_hand_jpos = normalized_jpos[:, LEFT_HAND_IDX] - normalized_jpos[:, LEFT_HAND_IDX[0:1]] 
		right_hand_jpos = normalized_jpos[:, RIGHT_HAND_IDX] - normalized_jpos[:, RIGHT_HAND_IDX[0:1]]
		left_global_rot6d = global_rot6d[:, LEFT_HAND_IDX]
		right_global_rot6d = global_rot6d[:, RIGHT_HAND_IDX]
		hand_motion = np.concatenate((
										left_hand_jpos.reshape(-1, 16*3),
										left_global_rot6d.reshape(-1, 16*6), # global_rot6d[:, 22*6:37*6],
										right_hand_jpos.reshape(-1, 16*3),
										right_global_rot6d.reshape(-1, 16*6),# global_rot6d[:, 37*6:]
										),
										axis=-1)

		max_motion_length = self.opt.max_motion_length
		if m_length >= self.opt.max_motion_length:
			body_motion = body_motion[0:max_motion_length]
			hand_motion = hand_motion[0:max_motion_length]
			m_length = len(body_motion)
		else:
			padding_len = max_motion_length - m_length
			D1 = body_motion.shape[1]
			D2 = hand_motion.shape[1]
			body_padding_zeros = np.zeros((padding_len, D1))
			hand_padding_zeros = np.zeros((padding_len, D2))
			body_motion = np.concatenate((body_motion, body_padding_zeros), axis=0)
			hand_motion = np.concatenate((hand_motion, hand_padding_zeros), axis=0)

		assert len(body_motion) == len(hand_motion) == max_motion_length

		return hand_text, body_motion, hand_motion, m_length
