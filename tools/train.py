import os
from os.path import join as pjoin
from omegaconf import OmegaConf

from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import MotionTransformer, ConstraintManager
from trainers import DDPMTrainer
from datasets import Text2MotionDataset

import torch
import torch.distributed as dist

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

		
if __name__ == '__main__':
	parser = TrainCompOptions()
	opt = parser.parse()

	## uncomment this to resume training
	# opt.is_continue = True
	
	opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
	opt.model_dir = pjoin(opt.save_root, 'model')
	opt.meta_dir = pjoin(opt.save_root, 'meta')
	opt.gen_dir = pjoin(opt.save_root, 'generated')
	opt.save_cm_dir = pjoin(opt.save_root, 'cm_data')

	os.makedirs(opt.model_dir, exist_ok=True)
	os.makedirs(opt.meta_dir, exist_ok=True)
	os.makedirs(opt.gen_dir, exist_ok=True)
	os.makedirs(opt.save_cm_dir, exist_ok=True)

	opt.device = torch.device("cuda")

	## dataset-specific parameters
	if opt.dataset_name == 'both57M':
		opt.data_root = '/home/these/DATA/Dataset/OMar/Hands Dataset/BOTH2HANDS/both2hands1/'
		opt.motion_dir = pjoin(opt.data_root, 'motions')
		opt.hand_text_dir = pjoin(opt.data_root, 'texts', 'hands')
		opt.body_text_dir = pjoin(opt.data_root, 'texts', 'body')
		opt.num_joints = 52
		opt.num_body_joints = 22
		opt.num_hand_joints = 16
		opt.joint_dim = 9
		opt.max_motion_length = 150
		opt.dim_pose = 2 * opt.num_hand_joints * opt.joint_dim
		opt.text_latent_dim = 768

	elif opt.dataset_name == 'motionx':
		opt.data_root = '/data1/omar/motion-x/t2m version/'
		opt.motion_dir = pjoin(opt.data_root, 'motions')
		opt.hand_text_dir = pjoin(opt.data_root, 'texts')
		opt.num_joints = 52
		opt.num_body_joints = 22
		opt.num_hand_joints = 16
		opt.joint_dim = 9
		opt.max_motion_length = 150
		opt.dim_pose = 2 * opt.num_hand_joints * opt.joint_dim
		opt.text_latent_dim = 768

	else:
		raise KeyError('Dataset Does Not Exist')

	## load training data
	train_split_file = pjoin(opt.data_root, 'train.txt')
	train_dataset = Text2MotionDataset(opt, train_split_file)

	## anatomical constraints manager
	cm = ConstraintManager(train_dataset, 
						   num_timesteps=opt.diffusion_steps, 
						   save_data_path=opt.save_cm_dir, 
						   device=opt.device)

	## build denoiser model
	model = build_models(opt)
	model = model.to(device)

	## train
	trainer = DDPMTrainer(opt, model, cm)
	trainer.train(train_dataset)
