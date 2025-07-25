import os
import torch
import numpy as np
import argparse
from os.path import join as pjoin

from utils.plot_script import *
from utils.get_opt import get_opt

from trainers import DDPMTrainer
from models import MotionTransformer
from utils.utils import *
from omegaconf import OmegaConf

from utils.plot_script import *
from config import *
		

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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--opt_path', type=str, help='Opt path')
	parser.add_argument('--text', type=str, default="", help='Text description for motion generation')
	parser.add_argument('--motion_length', type=int, default=60, help='Number of frames for motion generation')
	parser.add_argument('--fname', type=str, default="", help='Path to save 3D keypoints sequence')
	parser.add_argument('--gpu_id', type=int, default=-1, help="which gpu to use")
	args = parser.parse_args()
	
	device = torch.device('cuda:%d' % args.gpu_id if args.gpu_id != -1 else 'cpu')
	opt = get_opt(args.opt_path, device)
	opt.do_denoise = True

	opt.num_joints = 52
	opt.num_body_joints = 20
	opt.num_hand_joints = 16
	opt.joint_dim = 9
	opt.max_motion_length = 150
	opt.dim_pose = 2 * opt.num_hand_joints * opt.joint_dim
	opt.text_latent_dim = 768

	## load pretrained model	
	model = build_models(opt).to(device)
	trainer = DDPMTrainer(opt, encoder, None)
	trainer.load(pjoin(opt.model_dir, 'latest.tar'))
	trainer.eval_mode()
	trainer.to(opt.device)

	result_dict = {}
	with torch.no_grad():
		if args.motion_length != -1:
			caption = [args.hand_text]
			m_lens = torch.LongTensor([opt.max_motion_length]).to(device)
			gen_motions = trainer.generate(caption, m_lens, opt.dim_pose)
			gen_motion = gen_motions[0].cpu().numpy()
			print('motion shape:', gen_motion.shape)

			##
			positions = gen_motion[:, :52*3].reshape(gen_motion.shape[0], 52, 3)
			positions = de_normalize_jpos_min_max(positions)
			positions = motion_temporal_filter(positions, sigma=1)

			save_dict = {}
			save_dict["motion"] = gen_motion
			save_dict["positions"] = positions
			save_dict["seq_len"] = m_lens.cpu().numpy()
			save_dict["text"] = caption[0]

			np.save(pjoin(opt.gen_dir, args.fname), save_dict)

			save_anim_path = pjoin(opt.gen_dir, args.fname.replace('.npy', '.gif'))
			plot_3d_motion(save_anim_path, kinematic_chain, positions[:, 1:], caption[0], figsize=(10, 10), fps=120, radius=1.0)

			print('The animation is saved at:' + save_anim_path + ' !')
