#############################################
import os
import numpy as np
from os.path import join as pjoin
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import math


kinematic_chain = [
	
	[0, 1, 2, 3],     # Right_Hand → Right_Thumb_MCP → Right_Thumb_PIP → Right_Thumb_DIP
	[0, 4, 5, 6],     # Right_Hand → Right_Index_MCP → Right_Index_PIP → Right_Index_DIP
	[0, 7, 8, 9],     # Right_Hand → Right_Middle_MCP → Right_Middle_PIP → Right_Middle_DIP
	[0, 10, 11, 12],     # Right_Hand → Right_Ring_MCP → Right_Ring_PIP → Right_Ring_DIP
	[0, 13, 14, 15]      # Right_Hand → Right_Pinky_MCP → Right_Pinky_PIP → Right_Pinky_DIP
]

joint_parents_map = {}
for l in kinematic_chain:
	for i in range(1, len(l)-1):
		j1 = l[i] 
		j2 = l[i+1]
		joint_parents_map[j2] = j1

def calc_bone_len(p1, p2):
	return np.linalg.norm(p1 - p2, axis=-1)
	
def calc_min_max_bones(poses):
	
	bones_mins = np.ones(len(joint_parents_map)) * np.inf
	bones_maxs = np.zeros(len(joint_parents_map))
	for pose in tqdm(poses):
		bones_len = [calc_bone_len(pose[joint_parents_map[child], :3], pose[child, :3]) for child in joint_parents_map]
			
		bones_mins = np.minimum(bones_mins, bones_len)
		bones_maxs = np.maximum(bones_maxs, bones_len)            
			
	return bones_mins, bones_maxs


def calc_min_max_rots(poses):    
	rots_mins = np.ones((len(joint_parents_map), 6)) * np.inf
	rots_maxs = np.zeros((len(joint_parents_map), 6))
	
	for pose in tqdm(poses):
		rots = [pose[child, 3:] for child in joint_parents_map]
			
		rots_mins = np.minimum(rots_mins, rots)
		rots_maxs = np.maximum(rots_maxs, rots)            
			
	return rots_mins, rots_maxs


class AdaptiveGuidanceWeights:
	def __init__(self, total_steps=1000, schedule_type='linear', 
				 max_lambda=0.9, min_lambda=0.1):
		"""
		Adaptive guidance weight scheduler
		Args:
			total_steps: Total diffusion timesteps (T)
			schedule_type: 'cosine', 'linear', or 'exponential'
			max_lambda: Maximum guidance weight (at t=T)
			min_lambda: Minimum guidance weight (at t=0)
		"""
		self.total_steps = total_steps
		self.schedule_type = schedule_type
		self.max_lambda = max_lambda
		self.min_lambda = min_lambda
		
		if schedule_type == 'exponential':
			self.decay_rate = math.log(min_lambda/max_lambda)/total_steps

	def __call__(self, t):
		"""Compute weight for given timestep(s)
		Args:
			t: Tensor of timesteps [B,] (values from 0 to T-1)
		Returns:
			lambda_t: Tensor of weights [B,]
		"""
		progress = t.float() / (self.total_steps - 1)
		
		if self.schedule_type == 'linear':
			return self.max_lambda - (self.max_lambda - self.min_lambda) * progress
		
		elif self.schedule_type == 'cosine':
			return self.min_lambda + 0.5*(self.max_lambda - self.min_lambda)*(
				1 + torch.cos(math.pi * progress))
		
		elif self.schedule_type == 'exponential':
			return self.max_lambda * torch.exp(self.decay_rate * (self.total_steps - 1 - t))
		
		else:
			raise ValueError(f"Unknown schedule: {self.schedule_type}")



class ConstraintManager:
  def __init__(self, dataset, num_timesteps, save_data_path=None, device='cuda'):
	  """
	  Args:
		  joint_parents: List defining parent index for each joint
		  bone_limits: Tensor [num_bones, 2] containing (min, max) lengths
		  angle_hulls: List of ConvexHull objects for joint angle constraints
	  """
	  self.num_joints = 16
	  self.num_timesteps = num_timesteps
	  self.device = device
	  self.joint_parents = joint_parents_map
	  self.children = list(joint_parents_map.keys())
	  self.parents = list(joint_parents_map.values())
	  bone_limits, rotations_limits = self._get_limits(dataset, save_data_path)
	  self.bone_limits = torch.from_numpy(bone_limits).unsqueeze(1).unsqueeze(1).to(device)
	  self.rotations_limits = torch.from_numpy(rotations_limits).unsqueeze(1).to(device)		

	  self.guidance_scheduler = AdaptiveGuidanceWeights(
		  total_steps=self.num_timesteps,
		  schedule_type='exponential',
		  max_lambda=1.2,
		  min_lambda=0.5
	  )
		
  def _get_limits(self, dataset, save_path=None):
	  bones_mins = np.ones(len(self.joint_parents)) * np.inf
	  bones_maxs = np.ones(len(self.joint_parents)) * -np.inf

	  rots_mins = np.ones((self.num_joints * 6)) * np.inf
	  rots_maxs = np.ones((self.num_joints * 6)) * -np.inf

	  self.mins = np.ones(self.num_joints*9) * np.inf
	  self.maxs = np.ones(self.num_joints*9) * -np.inf

	  for i in tqdm(range(dataset.real_len()), desc='calc data limits....'):
	  # for i in tqdm(range(100), desc='calc data limits....'):
		  _, _, motion, _ = dataset.__getitem__(i)
		  T, d = motion.shape
			
		  positions_left = motion[:, :self.num_joints*3].reshape(T, self.num_joints, -1)
		  rotations_left = motion[:, self.num_joints*3:self.num_joints*9]

		  positions_right = motion[:, self.num_joints*9:self.num_joints*9+self.num_joints*3].reshape(T, self.num_joints, -1)
		  rotations_right = motion[:, self.num_joints*9+self.num_joints*3:]
		  
		  for pose in positions_left:
			  vec = pose[self.children] - pose[self.parents]
			  curr_len = np.linalg.norm(vec, axis=-1)              
			  bones_mins = np.minimum(bones_mins, curr_len)
			  bones_maxs = np.maximum(bones_maxs, curr_len)   

		  for rot in rotations_left:                   
			  rots_mins = np.minimum(rots_mins, rot)
			  rots_maxs = np.maximum(rots_maxs, rot)  

		  for pose in positions_right:
			  vec = pose[self.children] - pose[self.parents]
			  curr_len = np.linalg.norm(vec, axis=-1)              
			  bones_mins = np.minimum(bones_mins, curr_len)
			  bones_maxs = np.maximum(bones_maxs, curr_len)   

		  for rot in rotations_right:                   
			  rots_mins = np.minimum(rots_mins, rot)
			  rots_maxs = np.maximum(rots_maxs, rot)  

	  bone_limits = np.stack((bones_mins, bones_maxs), axis=0)
	  rotations_limits = np.stack((rots_mins, rots_maxs), axis=0) 

	  if save_path is not None:
		  np.save(pjoin(save_path, 'lengths_limits.npy'), bone_limits)
		  np.save(pjoin(save_path, 'rotations_limits.npy'), rotations_limits)
		  np.save(pjoin(save_path, 'parent_map.npy'), self.joint_parents)

	  return bone_limits, rotations_limits


  def project_to_feasible(self, x, t):
	  """Hierarchical projection through kinematic chain"""
	  x_proj = x.clone()
	  B, T, _ = x.shape
	  x_proj = x_proj.view(B, T, self.num_joints, -1)
	  B, T, J, D = x_proj.shape
		
	  vec = x_proj[..., self.children, :3] - x_proj[..., self.parents, :3]
	  curr_len = torch.norm(vec, dim=-1, keepdim=True)

	  ## scale postions
	  lambda_t = self.guidance_scheduler(t).view(-1, 1, 1, 1)

	  # Soft clamping with gradient flow
	  min_len, max_len = self.bone_limits[0].unsqueeze(-1), self.bone_limits[1].unsqueeze(-1)
	  clamped_len = lambda_t * max_len * torch.sigmoid(
	      (curr_len / max_len)
	  )
	  # Maintain gradient flow using straight-through estimator
	  vec = (clamped_len / curr_len) * vec
	  
	  x_proj[..., self.children,:3] = x_proj[..., self.parents,:3] + vec.float()
	  x_proj = x_proj.view(B, T, -1)

	  return x_proj

  def _project_angles(self, angles, joint_idx):
	  """Convex hull projection using precomputed planes"""
	  A, b = self.hull_normals[joint_idx]
	  violations = (A @ angles) - b
	  mask = violations > 0
		
	  if not mask.any():
		  return angles  # Already inside hull
		
	  # Find nearest hull point using quadratic programming
	  diff = self.hull_vertices[joint_idx] - angles
	  dists = torch.norm(diff, dim=1)
	  return self.hull_vertices[joint_idx][torch.argmin(dists)]

  def compute_violation(self, x):
	  """Differentiable constraint violation metric"""
	  bone_viol = self._bone_violation(x)
	  rots_viol = self._rotation_violation(x)
	  return bone_viol + 0.5 * rots_viol  # Weighted sum

  def _bone_violation(self, x):
	  """Bone length violation energy"""
	  x_proj = x.clone()
	  B, T, _ = x.shape
	  x_proj = x_proj.view(B, T, self.num_joints, -1)
	  B, T, J, D = x_proj.shape
		
	  ## violation
	  viol = 0.0
	  vec = x_proj[..., self.parents, :3] - x_proj[..., self.children, :3]
	  curr_len = torch.norm(vec, dim=-1)
		
	  viol += F.relu(curr_len - self.bone_limits[1]).mean()
	  viol += F.relu(self.bone_limits[0] - curr_len).mean()
	  return viol / (J-1)  # Normalize by bone count

  def _rotation_violation(self, x):
	  """Angle hull violation energy"""
	  # angles = x[..., 3:].view(-1, 2)
	  viol = 0.0
		
	  # for i, a in enumerate(angles):
	  #     joint_idx = i % len(self.hull_normals)
	  #     A, b = self.hull_normals[joint_idx]
	  #     dist = torch.max(A @ a - b)
	  #     viol += F.relu(dist)
	  rots = x[..., self.num_joints*3:]
	  min_rots = self.rotations_limits[0]
	  max_rots = self.rotations_limits[1]
	  viol += F.relu(rots - max_rots).mean()
	  viol += F.relu(min_rots - rots).mean()
			
	  return viol / rots.shape[0]
	
  def calc_rots(self, x):
	  rots = [x[:, :, child, 3:]  for child in self.joint_parents]
	  return torch.stack(rots, dim=1)

  def generate_in_range(self, x) -> torch.Tensor:
	  """
	  Generate random values where each element x[b,t,d] is uniformly sampled 
	  between MinX[b,t,d] and MaxX[b,t,d]
		
	  Args:
		  MinX: Tensor of minimum values (shape B, T, d)
		  MaxX: Tensor of maximum values (shape B, T, d)
		
	  Returns:
		  Random tensor of shape (B, T, d) with values in [MinX, MaxX)
	  """     
	  # Generate random values in [0, 1) with matching shape/device
	  rand = torch.randn_like(x, device=x.device)
	  # Scale and shift to desired range
	  return self.mins + (self.maxs - self.mins) * rand