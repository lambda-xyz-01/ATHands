import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin
import sys
from tqdm import tqdm

from utils.utils import *

from datasets import Text2MotionDataset
from torch.utils.data import DataLoader
from utils.get_opt import get_opt

from trainers import DDPMTrainer
from models import MotionTransformer


def build_models(opt):
	'''Motion Transfomer Model'''
	model = MotionTransformer(
		num_joints=opt.num_hand_joints,
		input_feats=opt.dim_pose,
		num_frames=opt.max_motion_length,
		num_layers=opt.num_layers,
		text_latent_dim=opt.text_latent_dim,
		latent_dim=opt.latent_dim,
		no_clip=opt.no_clip,
		no_eff=opt.no_eff)
	return model


def de_normalize_jpos_min_max(normalized_jpos, min_jpos, max_jpos):
	normalized_jpos = (normalized_jpos + 1) * 0.5 # [0, 1] range

	diff = max_jpos - min_jpos
	diff[diff == 0] = 1e-8  # set very small value to prevent division by zero

	de_jpos = normalized_jpos * diff + min_jpos
	return de_jpos # T X N X 3


def generate_val_res(evl_folder, model, val_batch, batch_idx, val_epoch_counter, opt): 
	## create folder for generated data
	sub_folder = os.path.join(evl_folder,str(val_epoch_counter))
	os.makedirs(sub_folder,exist_ok=True)
	with torch.no_grad():

		## val data
		hand_text, body_motion, hand_motion, m_length = val_batch
		m_length = m_length.to(opt.device)
		body_motion = body_motion.to(opt.device).float()

		## generate motion using trained model
		pred_hand_motion = trainer.generate(hand_text, body_motion, m_length, opt.dim_pose)
		pred_hand_motion = np.array([p.cpu().numpy() for p in pred_hand_motion])

		T = pred_hand_motion.shape[1]
		body_motion = body_motion[:, :T].detach().cpu().numpy()		
		hand_motion = hand_motion[:, :T].detach().cpu().numpy()

		## concatenate gt/gen hands motions with body motion
		gen_motion = np.concatenate((body_motion[..., :opt.num_body_joints*3],
									  pred_hand_motion[..., 3:opt.num_hand_joints*3],
									  pred_hand_motion[..., opt.num_hand_joints*9+3:opt.num_hand_joints*9+opt.num_hand_joints*3],
									  body_motion[..., opt.num_body_joints*3:],
									  pred_hand_motion[..., opt.num_hand_joints*3+6:opt.num_hand_joints*9],
									  pred_hand_motion[..., 2*opt.num_hand_joints*3+opt.num_hand_joints*6+6:],
									  ),
									axis=-1)

		gt_motion = np.concatenate((body_motion[..., :opt.num_body_joints*3],
									  hand_motion[..., 3:opt.num_hand_joints*3],
									  hand_motion[..., opt.num_hand_joints*9+3:opt.num_hand_joints*9+opt.num_hand_joints*3],
									  body_motion[..., opt.num_body_joints*3:],
									  hand_motion[..., opt.num_hand_joints*3+6:opt.num_hand_joints*9],
									  hand_motion[..., 2*opt.num_hand_joints*3+opt.num_hand_joints*6+6:],
									  ),
									axis=-1)

		## get text embedding
		hand_text_emb = trainer.encoder.encode_text(hand_text, opt.device)[1]

		## save gt/gen data
		save_dict = {}
		save_dict["gt_motion"] = gt_motion
		save_dict["seq_len"] = m_length.cpu().numpy()
		save_dict["hand_text_embed"] = hand_text_emb.cpu().numpy()
		save_dict["hand_text"] = hand_text
		save_dict["gen_motion"] = gen_motion
		
		save_path = os.path.join(sub_folder, str(batch_idx)+".npy")
		np.save(save_path, save_dict)


if __name__ == '__main__':
	
	opt_path = sys.argv[1]
	dataset_opt_path = opt_path
	try:
		device_id = int(sys.argv[2])
	except:
		device_id = 0
	device = torch.device('cuda:%d' % device_id if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(device_id)

	opt = get_opt(dataset_opt_path, device)
	opt.do_denoise = True

	opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
	opt.model_dir = pjoin(opt.save_root, 'model')
	opt.meta_dir = pjoin(opt.save_root, 'meta')
	opt.gen_dir = pjoin(opt.save_root, 'generated_motions')
	
	opt.data_root = '/data1/omar/BOTH2HANDS/both2hands1/'
	opt.motion_dir = pjoin(opt.data_root, 'motions')
	opt.text_dir = pjoin(opt.data_root, 'texts', 'hands')
	opt.num_joints = 52
	opt.num_body_joints = 22
	opt.num_hand_joints = 16
	opt.joint_dim = 9
	opt.max_motion_length = 150
	opt.dim_pose = 2 * opt.num_hand_joints * opt.joint_dim
	opt.text_latent_dim = 768

	evl_folder = opt.gen_dir
	os.makedirs(evl_folder, exist_ok=True)

	batch_size = 32
	test_split_file = pjoin(opt.data_root, 'test.txt')
	test_dataset = Text2MotionDataset(opt, test_split_file)
	test_loader = DataLoader(
						test_dataset,
						batch_size=batch_size)

	model = build_models(opt).to(device)
	trainer = DDPMTrainer(opt, model, None)
	epoch, _ = trainer.load(pjoin(opt.model_dir, 'latest.tar'))
	print('epoch:', epoch)

	trainer.eval_mode()
	trainer.to(opt.device)

	opt.validate_times = 10
	val_epoch_counter = 0
	
	for i in range(opt.validate_times):
		with torch.no_grad():
			for batch_idx, data_batch in tqdm(enumerate(test_loader)):
				generate_val_res(evl_folder, trainer, data_batch, batch_idx, val_epoch_counter, opt)
			val_epoch_counter += 1

				

