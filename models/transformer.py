"""
Copyright 2021 S-Lab
"""
import torch
import torch.nn.functional as F
from torch import layer_norm, nn
import numpy as np
import clip
from einops import rearrange
import math
import re

from config import *


def timestep_embedding(timesteps, dim, max_period=10000):
	"""
	Create sinusoidal timestep embeddings.
	:param timesteps: a 1-D Tensor of N indices, one per batch element.
					  These may be fractional.
	:param dim: the dimension of the output.
	:param max_period: controls the minimum frequency of the embeddings.
	:return: an [N x dim] Tensor of positional embeddings.
	"""
	half = dim // 2
	freqs = torch.exp(
		-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
	).to(device=timesteps.device)
	args = timesteps[:, None].float() * freqs[None]
	embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
	if dim % 2:
		embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
	return embedding


def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad for all the networks.

	Args:
		nets (nn.Module | list[nn.Module]): A list of networks or a single
			network.
		requires_grad (bool): Whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


def zero_module(module):
	"""
	Zero out the parameters of a module and return it.
	"""
	for p in module.parameters():
		p.detach().zero_()
	return module


class StylizationBlock(nn.Module):

	def __init__(self, latent_dim, time_embed_dim, dropout):
		super().__init__()
		self.emb_layers = nn.Sequential(
			nn.SiLU(),
			nn.Linear(time_embed_dim, 2 * latent_dim),
		)
		self.norm = nn.LayerNorm(latent_dim)
		self.out_layers = nn.Sequential(
			nn.SiLU(),
			nn.Dropout(p=dropout),
			zero_module(nn.Linear(latent_dim, latent_dim)),
		)

	def forward(self, h, emb):
		"""
		h: B, T, D
		emb: B, D
		"""
		# B, 1, 2D
		emb_out = self.emb_layers(emb).unsqueeze(1)
		# scale: B, 1, D / shift: B, 1, D
		scale, shift = torch.chunk(emb_out, 2, dim=2)
		h = self.norm(h) * (1 + scale) + shift
		h = self.out_layers(h)
		return h


class LinearTemporalSelfAttention(nn.Module):

	def __init__(self, seq_len, latent_dim, num_head, dropout, time_embed_dim):
		super().__init__()
		self.num_head = num_head
		self.norm = nn.LayerNorm(latent_dim)
		self.query = nn.Linear(latent_dim, latent_dim)
		self.key = nn.Linear(latent_dim, latent_dim)
		self.value = nn.Linear(latent_dim, latent_dim)
		self.dropout = nn.Dropout(dropout)
		self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
	
	def forward(self, x, emb, src_mask):
		"""
		x: B, T, D
		"""
		B, T, D = x.shape
		H = self.num_head
		# B, T, D
		query = self.query(self.norm(x))
		# B, T, D
		key = (self.key(self.norm(x)) + (1 - src_mask) * -1000000)
		query = F.softmax(query.view(B, T, H, -1), dim=-1)
		key = F.softmax(key.view(B, T, H, -1), dim=1)
		# B, T, H, HD
		value = (self.value(self.norm(x)) * src_mask).view(B, T, H, -1)
		# B, H, HD, HD
		attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
		y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
		y = x + self.proj_out(y, emb)
		return y


class LinearTemporalCrossAttention(nn.Module):
	def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
		super().__init__()
		self.num_head = num_head
		self.norm = nn.LayerNorm(latent_dim)
		self.text_norm = nn.LayerNorm(text_latent_dim)
		self.query = nn.Linear(latent_dim, latent_dim)
		self.key = nn.Linear(text_latent_dim, latent_dim)
		self.value = nn.Linear(text_latent_dim, latent_dim)
		self.dropout = nn.Dropout(dropout)
		self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)
	
	def forward(self, x, xf, emb, src_mask=None):
		"""
		x: B, T, D
		xf: B, N, L
		src_mask: B, N (optional mask for xf)
		"""
		B, T, D = x.shape
		N = xf.shape[1]
		H = self.num_head
		
		# Compute query, key, value
		query = self.query(self.norm(x))  # (B, T, D)
		key = self.key(self.text_norm(xf))  # (B, N, D)
		query = F.softmax(query.view(B, T, H, -1), dim=-1)
		if src_mask is not None:
			key = key + (1 - src_mask) * -1000000  # Mask padding

		key = F.softmax(key.view(B, N, H, -1), dim=1)
		value = self.value(self.text_norm(xf))  # (B, N, D)
		# Apply mask to key and value (if provided)
		if src_mask is not None:
			value = value * src_mask  # Zero out padded values
		
		# Reshape and compute attention
		value = value.view(B, N, H, -1)
		
		# B, H, HD, HD
		attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
		y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
		y = x + self.proj_out(y, emb)
		return y

class FFN(nn.Module):

	def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim):
		super().__init__()
		self.linear1 = nn.Linear(latent_dim, ffn_dim)
		self.linear2 = zero_module(nn.Linear(ffn_dim, latent_dim))
		self.activation = nn.GELU()
		self.dropout = nn.Dropout(dropout)
		self.proj_out = StylizationBlock(latent_dim, time_embed_dim, dropout)

	def forward(self, x, emb):
		y = self.linear2(self.dropout(self.activation(self.linear1(x))))
		y = x + self.proj_out(y, emb)
		return y
	

class LinearTemporalDiffusionTransformerDecoderLayer(nn.Module):

	def __init__(self,
				 seq_len=60,
				 latent_dim=32,
				 text_latent_dim=512,
				 time_embed_dim=128,
				 ffn_dim=256,
				 num_head=4,
				 dropout=0.1):
		super().__init__()
		self.sa_block = LinearTemporalSelfAttention(seq_len, latent_dim, num_head, dropout, time_embed_dim)
		self.text_ca_block = LinearTemporalCrossAttention(seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim)
		self.body_ca_block = LinearTemporalCrossAttention(seq_len, latent_dim, latent_dim//2, num_head, dropout, time_embed_dim)
		self.ffn = FFN(latent_dim, ffn_dim, dropout, time_embed_dim)

	def forward(self, x, xf, emb, src_mask, body_emb):
		x = self.sa_block(x, emb, src_mask)
		x = self.body_ca_block(x, body_emb, emb, src_mask=src_mask)
		x = self.text_ca_block(x, xf, emb)
		x = self.ffn(x, emb)
		return x


class SinusoidalPosEmb(nn.Module):
	def __init__(self, dim):
		super().__init__()
		self.dim = dim

	def forward(self, x):
		device = x.device
		half_dim = self.dim // 2
		emb = math.log(10000) / (half_dim - 1)
		emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
		emb = x[:, None] * emb[None, :]
		emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
		return emb


class KinematicAwareTransformer(nn.Module):
	def __init__(self, num_frames, in_dim, out_dim, hand_parts, nhead, num_layers):
		super().__init__()
		self.hand_parts = hand_parts
		self.num_parts = len(hand_parts)
		self.max_joints = max(len(part) for part in hand_parts)
		self.num_joints = max([max(fig) for fig in hand_parts]) + 1
		
		self.joint_parents = dict()
		for finger_ids in hand_parts:
			for i in range(1, len(finger_ids)):
				self.joint_parents[finger_ids[i]] = finger_ids[i-1]

		# Input projection layer
		latent_dim = 128
		self.input_proj = nn.Linear(in_dim, latent_dim)

		# Positional encodings (time + joints)
		self.time_pe = nn.Parameter(torch.randn(num_frames, latent_dim))
		self.joint_pe = nn.Parameter(torch.randn(self.num_parts, latent_dim))

		# Transformer encoder
		encoder_layer = nn.TransformerEncoderLayer(
			d_model=latent_dim,
			nhead=nhead,
			dim_feedforward=4*latent_dim,  
			dropout=0.1,
			activation='gelu'
		)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
		
		self.to_dec = nn.Linear(self.num_parts*latent_dim, out_dim)

		self._init_weights()

	def _init_weights(self):
		"""Initialize parameters"""
		for p in self.parameters():
			if p.dim() > 1:
				nn.init.xavier_uniform_(p)
	
	def _compute_depths(self):
		depths = []
		for part in self.hand_parts:
			depths.append(0)
			for i in range(1, len(part[1:])+1):
			   depths.append(i)
		return torch.tensor(depths)

	def _relpos(self, x):
		relpos = torch.zeros_like(x)
		for j in range(self.num_joints):
			if j in self.joint_parents:
				parent = self.joint_parents[j]
				relpos[:, :, j] = x[:, :, j] - x[:, :, parent]
			else:
				relpos[:, :, j] = -1 * torch.ones_like(x[:, :, 0])
		return relpos
				
	def group_fingers(self, x):
		"""
		Args:
			x: Input tensor of shape (B, T, num_joints, joint_dim)
		Returns:
			Tensor of shape (B, T, total_padded_dim) where:
			total_padded_dim = num_parts * (max_joints_per_part * joint_dim)
		"""
		B, T, _, D = x.shape
		device = x.device
		
		grouped_parts = []
		
		for part_indices in self.hand_parts:
			# Extract joints for current body part
			part_joints = x[:, :, part_indices, :]  # (B, T, num_joints_in_part, D)
			num_joints = len(part_indices)
			
			# Create padded tensor
			padded_part = torch.zeros(B, T, self.max_joints, D, device=device)
			padded_part[:, :, :num_joints, :] = part_joints
			
			# Flatten joints and features
			flattened = padded_part.view(B, T, -1)  # (B, T, max_joints*D)
			grouped_parts.append(flattened.unsqueeze(2))
		
		# Concatenate all parts along feature dimension
		return torch.cat(grouped_parts, dim=2)

	def forward(self, x):
		device = x.device
		B, T, N, d = x.shape

		local_x = x[:, :, :, :3] - x[:, :, 0:1, :3]
		
		group_x = self.group_fingers(x)
		group_local_x = self.group_fingers(local_x)
		
		relpos = self._relpos(x[..., :3])
		relpos = self.group_fingers(relpos)

		x_cat = torch.cat((group_x, group_local_x, relpos), dim=-1)
				
		# Project input to d_model
		x_in = self.input_proj(x_cat)  # (B, T, N, d_model)

		# Generate positional encodings
		time_pos = self.time_pe[:T]  # (T, d_model)
		joint_pos = self.joint_pe  # (N, d_model)
		
		# Combine positional encodings
		pos_enc = time_pos.unsqueeze(1) + joint_pos.unsqueeze(0)  # (T, N, d_model)
		pos_enc = pos_enc.flatten(0, 1)  # (T*N, d_model)

		# Prepare transformer input
		x_in = x_in.flatten(1, 2)  # (B, T*N, d_model)
		x_in = x_in + pos_enc.unsqueeze(0)  # Add positional encoding

		# Transformer expects (S, B, d_model)
		x_in = x_in.permute(1, 0, 2)  # (T*N, B, d_model)

		# Transformer encoder
		x_enc = self.transformer(x_in)  # (T*N, B, d_model)

		# Pool over sequence dimension
		x_enc = x_enc.permute(1, 0, 2)  # (B, T*N, d_model)
		x_enc = x_enc.reshape(B, T, -1)
		
		x_enc = self.to_dec(x_enc)

		return x_enc

class MotionTransformer(nn.Module):
	def __init__(self,
				 hand_dim,
				 body_dim,
				 num_hand_joints,
				 num_frames=240,
				 latent_dim=512,
				 ff_size=1024,
				 num_layers=8,
				 num_heads=8,
				 dropout=0,
				 text_latent_dim=512,
				 no_eff=False,
				 **kargs):
		super().__init__()
		
		self.num_frames = num_frames
		self.num_joints = num_hand_joints
		self.latent_dim = latent_dim
		self.ff_size = ff_size
		self.num_layers = num_layers
		self.num_heads = num_heads
		self.dropout = dropout
		self.input_feats = input_feats
		self.time_embed_dim = latent_dim * 4
		self.sequence_embedding = nn.Parameter(torch.randn(num_frames, latent_dim))
	
		self.encoder = KinematicAwareTransformer(num_frames=num_frames, 
											  in_dim=4*9+4*3+4*3,
											  out_dim=latent_dim,
											  hand_parts=HANDS_FINGERS_IDS,
											  nhead=8, 
											  num_layers=4)

		self.body_emb = nn.Linear(body_dim, latent_dim)
	
		self.temporal_decoder_blocks = nn.ModuleList()
		for i in range(num_layers):
			self.temporal_decoder_blocks.append(
				LinearTemporalDiffusionTransformerDecoderLayer(
					seq_len=num_frames,
					latent_dim=2*latent_dim,
					text_latent_dim=text_latent_dim,
					time_embed_dim=self.time_embed_dim,
					ffn_dim=ff_size,
					num_head=num_heads,
					dropout=dropout
				)
			)
	
		# For noise level t embedding
		self.time_embed = nn.Sequential(
			nn.Linear(self.latent_dim, self.time_embed_dim),
			nn.SiLU(),
			nn.Linear(self.time_embed_dim, self.time_embed_dim),
		)
	
		self.init_text_encoder()
	
		self.out = zero_module(nn.Linear(2*latent_dim, 2*hand_dim))
	
	def init_text_encoder(self):
		self.clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False)
		set_requires_grad(self.clip_model, False)
		self.clip_training = "text_"
		textTransEncoderLayer = nn.TransformerEncoderLayer(
			d_model=768,
			nhead=4,
			dim_feedforward=1024,
			dropout=0.1,
			activation="gelu",
			batch_first=True)
		self.textTransEncoder = nn.TransformerEncoder(
			textTransEncoderLayer,
			num_layers=4)
		self.text_ln = nn.LayerNorm(768)
		self.cond_embeding = nn.Linear(768, 4*self.latent_dim)
	
	def encode_text(self, raw_text, device):
		##clip
		text = clip.tokenize(raw_text, truncate=True).to(device)
		x = self.clip_model.token_embedding(text).type(self.clip_model.dtype)  # [batch_size, n_ctx, d_model]
		x = x + self.clip_model.positional_embedding.type(self.clip_model.dtype)
		x = x.permute(1, 0, 2)  # NLD -> LND
		x = self.clip_model.transformer(x)
		x = x.permute(1, 0, 2)
		x = self.clip_model.ln_final(x).type(self.clip_model.dtype)
	
		x = self.textTransEncoder(x)
		x = self.text_ln(x)
		xf_out = x[torch.arange(x.shape[0]), text.argmax(dim=-1)].unsqueeze(1)

		xf_proj = self.cond_embeding(xf_out).squeeze()
		
		return xf_proj, xf_out
	
	def generate_src_mask(self, T, length):
		B = len(length)
		src_mask = torch.ones(B, T)
		for i in range(B):
			for j in range(length[i], T):
				src_mask[i, j] = 0
		return src_mask
	
	def split_hands(self, x):
		b, t, _ = x.shape
		x_left = torch.cat((x[..., :self.num_joints*3].reshape(b, t, self.num_joints, 3), 
							x[..., self.num_joints*3:self.num_joints*9].reshape(b, t, self.num_joints, 6)), 
						dim=-1)
		x_right = torch.cat((x[..., self.num_joints*9:self.num_joints*9+self.num_joints*3].reshape(b, t, self.num_joints, 3), 
							x[..., self.num_joints*9+self.num_joints*3:].reshape(b, t, self.num_joints, 6)),
						dim=-1)
	
		return x_left, x_right
	
	def forward(self, x, timesteps, length=None, text_cond=None, body_cond=None):
		"""
		x: B, T, D
		"""
		device = x.device
		B, T = x.shape[0], x.shape[1]
		
		emb = self.time_embed(timestep_embedding(timesteps, self.latent_dim))
		xf_proj, xf_out = self.encode_text(text_cond, device)
		emb = emb + xf_proj
		
		x_left, x_right = self.split_hands(x)
		h_left = self.encoder(x_left)
		h_right = self.encoder(x_right)

		body_emb = self.body_emb(body_cond)
		body_emb = body_emb[:, :T, :] + self.sequence_embedding.unsqueeze(0)[:, :T, :]

		h_left = h_left + self.sequence_embedding.unsqueeze(0)[:, :T, :]
		h_right = h_right + self.sequence_embedding.unsqueeze(0)[:, :T, :]
		
		h = torch.cat((h_left, h_right), dim=-1)
		
		src_mask = self.generate_src_mask(T, length).to(device).unsqueeze(-1)
		for module in self.temporal_decoder_blocks:
			h = module(h, xf_out, emb, src_mask, body_emb=body_emb)

		out = self.out(h).view(B, T, -1).contiguous()
		
		return out

