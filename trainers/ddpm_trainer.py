import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
import time
from models.transformer import MotionTransformer
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from collections import OrderedDict
from utils.utils import print_current_loss
from os.path import join as pjoin
import codecs as cs
import torch.distributed as dist

from utils.utils import *
from datasets import build_dataloader

from models.gaussian_diffusion import (
	GaussianDiffusion,
	get_named_beta_schedule,
	create_named_schedule_sampler,
	ModelMeanType,
	ModelVarType,
	LossType
)


class DDPMTrainer(object):

	def __init__(self, args, encoder, cm):
		self.opt = args
		self.device = args.device
		self.encoder = encoder
		self.diffusion_steps = args.diffusion_steps
		sampler = 'uniform'
		beta_scheduler = 'linear'
		betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
		self.diffusion = GaussianDiffusion(
			betas=betas,
			model_mean_type=ModelMeanType.EPSILON, #ModelMeanType.START_X
			model_var_type=ModelVarType.FIXED_SMALL,
			loss_type=LossType.MSE,
			cm=cm
		)

		self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
		self.sampler_name = sampler

		self.cm = cm

		if args.is_train:
			self.mse_criterion = torch.nn.MSELoss(reduction='none') #
		self.to(self.device)

	@staticmethod
	def zero_grad(opt_list):
		for opt in opt_list:
			opt.zero_grad()

	@staticmethod
	def clip_norm(network_list):
		for network in network_list:
			clip_grad_norm_(network.parameters(), 0.5)

	@staticmethod
	def step(opt_list):
		for opt in opt_list:
			opt.step()

	def forward(self, batch_data, it, eval_mode=False):
		hand_caption, body_motion, hand_motion, m_lens = batch_data
		hand_motion = hand_motion.detach().to(self.device).float()
		body_motion = body_motion.detach().to(self.device).float()

		x_start = hand_motion
		B, T = x_start.shape[:2]
		cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
		t, _ = self.sampler.sample(B, x_start.device)

		output = self.diffusion.training_losses(
			model=self.encoder,
			x_start=x_start,
			t=t,
			it=it,
			cur_len=cur_len,
			model_kwargs={"text_cond": hand_caption,
						  "body_cond": body_motion,
						  "length": cur_len}
		)

		self.real_noise = output['target']
		self.fake_noise = output['pred']

		try:
			self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
		except:
			self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

	def generate_batch(self, text_cond, body_cond, m_lens, dim_pose):
		
		B = len(text_cond)
		T = min(m_lens.max(), self.encoder.num_frames)
		output = self.diffusion.p_sample_loop(
			self.encoder,
			(B, T, dim_pose),
			clip_denoised=False,
			progress=True,
			model_kwargs={
				'text_cond': text_cond,
				'body_cond': body_cond,
				'length': m_lens
			})
		return output
		
	def generate(self, text_cond, body_cond, m_lens, dim_pose, batch_size=1024):
		N = len(text_cond)
		cur_idx = 0
		self.encoder.eval()
		all_output = [None] * N
		while cur_idx < N:
			batch_end = min(cur_idx + batch_size, N)
			batch_text_cond = text_cond[cur_idx:batch_end]
			batch_body_cond = body_cond[cur_idx:batch_end]
			batch_m_lens = m_lens[cur_idx:batch_end]
			output = self.generate_batch(batch_text_cond, batch_body_cond, batch_m_lens, dim_pose)
			all_output[cur_idx:batch_end] = output
			cur_idx = batch_end
		return all_output

	def backward_G(self):        
		# Calculate the motion reconstruction loss as before
		
		hand_loss = self.mse_criterion(self.fake_noise, self.real_noise).mean(dim=-1)
		hand_loss = (hand_loss * self.src_mask).sum() / self.src_mask.sum()

		self.total_loss = hand_loss
		
		loss_logs = OrderedDict()
		
		# Log individual and total losses
		loss_logs['hand_loss'] = hand_loss.item()
		loss_logs['total_loss'] = self.total_loss.item()
		
		return loss_logs
	
	def update(self):
		# Zero out gradients
		self.zero_grad([self.opt_encoder])
		
		# Compute backward pass and get logs
		loss_logs = self.backward_G()
		
		# Backpropagate the total loss
		self.total_loss.backward()
		
		# Clip gradients and update optimizer
		self.clip_norm([self.encoder])
		self.step([self.opt_encoder])
		
		return loss_logs

	def to(self, device):
		if self.opt.is_train:
			self.mse_criterion.to(device)
		self.encoder = self.encoder.to(device)

	def train_mode(self):
		self.encoder.train()

	def eval_mode(self):
		self.encoder.eval()

	def save(self, file_name, ep, total_it):
		state = {
			'opt_encoder': self.opt_encoder.state_dict(),
			# 'scheduler': self.scheduler.state_dict(),
			'ep': ep,
			'total_it': total_it
		}
		try:
			state['encoder'] = self.encoder.module.state_dict()
		except:
			state['encoder'] = self.encoder.state_dict()
		torch.save(state, file_name)
		return

	def load(self, model_dir):
		checkpoint = torch.load(model_dir, map_location=self.device)
		if self.opt.is_train:
			self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
			# self.scheduler.load_state_dict(checkpoint['scheduler'])
		self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
		return checkpoint['ep'], checkpoint.get('total_it', 0)

	def validate(self, batch_data, it):
		hand_caption, body_motion, hand_motion, m_lens = batch_data
		hand_motion = hand_motion.detach().to(self.device).float()
		body_motion = body_motion.detach().to(self.device).float()

		x_start = hand_motion
		B, T = x_start.shape[:2]
		cur_len = torch.LongTensor([min(T, m_len) for m_len in  m_lens]).to(self.device)
		t, _ = self.sampler.sample(B, x_start.device)
		
		with torch.no_grad():
			output = self.diffusion.training_losses(
				model=self.encoder,
				x_start=x_start,
				t=t,
				it=it,
				cur_len=cur_len,
				model_kwargs={"text_cond": hand_caption,
							  "body_cond": body_motion,
							  "length": cur_len}
			)
	
			try:
				src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
			except:
				src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

			# Calculate the motion reconstruction loss as before
			hand_loss = self.mse_criterion(output['pred'], output['target']).mean(dim=-1)
			hand_loss = (hand_loss * src_mask).sum() / src_mask.sum()
			
			val_loss = hand_loss
			
			loss_logs = OrderedDict()
			
			# Log individual and total losses
			loss_logs['hand_loss'] = hand_loss.item()
			loss_logs['total_loss'] = val_loss.item()

			return loss_logs
		
			
	def train(self, train_dataset, val_dataset):
		rank = 0
		self.to(self.device)
		self.opt_encoder = optim.Adam(self.encoder.parameters(), 
										lr=self.opt.lr, )
										#weight_decay=0.0002)
		# self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt_encoder, 
		# 												T_max=20, 
		# 												eta_min=1e-6,
		# 												 last_epoch=-1)
		it = 0
		cur_epoch = 0
		if self.opt.is_continue:
			model_dir = pjoin(self.opt.model_dir, 'latest.tar')
			cur_epoch, it = self.load(model_dir)

		start_time = time.time()

		train_loader = build_dataloader(
			train_dataset,
			samples_per_gpu=self.opt.batch_size,
			drop_last=True,
			workers_per_gpu=4,
			shuffle=True,
			dist=self.opt.distributed,
			num_gpus=1)

		logs = OrderedDict()
		best_val = float('inf')
		for epoch in range(cur_epoch, self.opt.num_epochs):
			self.train_mode()
			train_loss = 0.0

			if epoch > 20:
				self.diffusion.cm = self.cm
				
			for i, batch_data in enumerate(tqdm(train_loader, desc='train')):
				self.forward(batch_data, it)

				train_log_dict = self.update()
				for k, v in train_log_dict.items():
					if k not in logs:
						logs[k] = v
					else:
						logs[k] += v

				train_loss = train_loss + train_log_dict['total_loss']
				
			train_loss = train_loss / len(train_loader)
			# val_loss = val_loss / len(val_loader)
			# val_body_loss = val_body_loss / len(val_loader)
			# val_hand_loss = val_hand_loss / len(val_loader)
			#val_bmc_loss = val_bmc_loss / len(val_loader)
			loss_log = {'train_loss': train_loss, 
						# 'val_loss': val_loss,
						# 'body_loss':val_body_loss,
						# 'hand_loss':val_hand_loss,
					   # 'bmc_loss': val_bmc_loss
					   }
			print_current_loss(start_time, epoch, loss_log, epoch, inner_iter=i)
			
			# if val_loss < best_val:
			# 	best_val = val_loss
			# 	self.save(pjoin(self.opt.model_dir, 'best.tar'), epoch, it)


			self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
				
			if epoch % 10 == 0 and epoch > 1:
				hand_caption, body_motion, hand_motion, m_lens = batch_data
				body_motion = body_motion.to(self.device).float()


				hand_caption = ['Circle left thumb Twice, other left fingers remain natural. Make a Fist with right hand.']
				pred_hand_motion = self.generate(hand_caption[0:1], body_motion[0:1], m_lens[0:1], self.opt.dim_pose)[0]
		
				body_motion = body_motion[0].detach().cpu().numpy()
				pred_hand_motion = pred_hand_motion.detach().cpu().numpy()
	
				pred_motion = np.concatenate((body_motion[:m_lens[0], :22*3],
											  pred_hand_motion[:, 3:16*3],
											  pred_hand_motion[:, 16*9+3:16*9+16*3],
											  body_motion[:m_lens[0], 22*3:],
											  pred_hand_motion[:, 16*3:16*9],
											  pred_hand_motion[:, 32*3+16*6:],
											  ),
											axis=-1)
	
				positions = train_dataset.de_normalize_jpos_min_max(pred_motion[:, :52*3].reshape(-1, 52, 3))
				pred_motion[..., :52*3] = positions.reshape(-1, 52 * 3)
				
				# pred_motion = np.concatenate((body_motion[:m_lens[0], :21*3],
				# 							  pred_hand_motion[:, 3:16*3],
				# 							  pred_hand_motion[:, 16*9+3:16*9+16*3],
				# 							  body_motion[:m_lens[0], 21*3:],
				# 							  pred_hand_motion[:, 16*3:16*9],
				# 							  pred_hand_motion[:, 32*3+16*6:],
				# 							  ),
				# 							axis=-1)
	
				# positions = train_dataset.de_normalize_jpos_min_max(pred_motion[:, :51*3].reshape(-1, 51, 3))
				# pred_motion[..., :51*3] = positions.reshape(-1, 51 * 3)
				
				
				positions = motion_temporal_filter(positions, sigma=1)
				save_dict = {}
				save_dict["pred_motion"] = pred_motion
				save_dict["pred_pos"] = positions
				save_dict["pred_hand_motion"] = pred_hand_motion
				save_dict["body_motion"] = body_motion
				save_dict["hand_motion"] = hand_motion[0].detach().cpu().numpy()
				# save_dict["joints"] = pred_joints
				save_dict["seq_len"] = m_lens[0].cpu().numpy()
				save_dict["hand_caption"] = hand_caption[0]
				# np.save(pjoin(self.opt.gen_dir, 'epoch_{}.npy'.format(epoch)), save_dict)
				np.save(pjoin(self.opt.gen_dir, 'generated_{}.npy'.format(epoch)), save_dict)
								

					





