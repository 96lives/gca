import torch
import MinkowskiEngine as ME
from torch.utils.tensorboard import SummaryWriter
from typing import List
from collections import defaultdict
from models.transition_model import TransitionModel
from utils.pad import unpack, get_gt_values
from utils.util import timeit, downsample
from utils.scheduler import InfusionScheduler
from utils.phase import Phase
from MinkowskiEngine import SparseTensor
from utils.visualization import (
	vis_2d_coords, tensors2dist_func_tensor_imgs, tensors2tensor_imgs
)
from utils.marching_cube import marching_cubes_sparse_voxel

class GCA(TransitionModel):
	name = 'gca'

	def __init__(self, config, writer: SummaryWriter):
		TransitionModel.__init__(self, config, writer)
		self.infusion_scheduler = InfusionScheduler(config)
		self.bce_loss = torch.nn.BCEWithLogitsLoss()

	@timeit
	def forward(self, x):
		'''
		Forward pass through sparse convolution network
		and unpack the output

		input:
			x: SparseTensor of
				coordinates with shape N x 3
				features with shape N x 1
		output:
			x_hat: SparseTensor of
				coordinates with shape M x 3
				features with shape M x k (parameter outputs)
		'''
		out_packed = self.backbone(x)
		out_unpacked = unpack(out_packed, self.shifts[:, 1:], self.out_dim)
		return out_unpacked

	@timeit
	def learn(
			self, data: dict,
			step: float, mode: str = 'train'
	) -> (dict, float):
		"""
		:param data: dict containing key, value pairs of
			- state_coord: Tensor containing coordinates of input voxels
			- state_feat: Tensor containing features of input voxels
			- query_point: Tensor of B x N x data_dim
			- dist: Tensor of B x N x data_dim
			- phase: List of phases for each data
		:param step: training step
		:param mode: mode of training
		:return:
			- next_step: dict containing same keys and values as parameter data
			- loss: float of the current step's lose
		"""

		s_coords = ME.utils.batched_coordinates(data['state_coord'])
		s_feats = torch.ones(s_coords.shape[0], 1)
		s = SparseTensor(
			features=s_feats,
			coordinates=s_coords,
			device=self.device,
		)

		y_coords = ME.utils.batched_coordinates(data['embedding_coord'])
		y_feats = torch.ones(y_coords.shape[0], 1)
		y = SparseTensor(
			features=y_feats,
			coordinates=y_coords,
			device=self.device,
		)
		# forward pass
		s_hat = self.forward(s)

		# compute loss
		losses = []
		one_hot_gt, y_pad_coords = get_gt_values(s_hat, y)

		phases = data['phase']
		batch_size = len(phases)
		feats = self.sample_feat(s_hat.F)
		infusion_rates = self.infusion_scheduler.sample(phases)

		s_next_feats = []
		s_next_coords = []
		for batch_idx, (infusion_rate, phase) in enumerate(zip(infusion_rates, phases)):
			# compute loss
			idx = s_hat.C[:, 0] == batch_idx
			s_hat_feat = s_hat.F[idx, :]
			losses.append(self.bce_loss(s_hat_feat.squeeze(1), one_hot_gt[batch_idx].float()))

			# infusion training
			feat = feats[idx]
			coord = s_hat.C[idx, :]
			infusion_idx = (torch.rand(feat.shape[0]) < infusion_rate)
			s_next_feat = torch.where(
				infusion_idx, one_hot_gt[batch_idx].float().cpu(), feat.cpu()
			)
			s_next_coords.append(coord[s_next_feat.bool(), 1:].cpu())
			s_next_feats.append(torch.ones(s_next_coords[batch_idx].shape[0], 1).cpu())

			# update_phases
			phases[batch_idx] = phase + 1
			completion_rate = one_hot_gt[batch_idx].sum().item() \
							  / float((y.C[:, 0] == batch_idx).sum().item())
			if completion_rate >= self.config['completion_rate']:
				if not phase.equilibrium_mode:
					phase.set_complete()
					self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]
			elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
				incomplete_key = 'phase/incomplete_cnt'
				self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
					len(self.scalar_summaries[incomplete_key]) != 0 else [1]

		loss = torch.stack(losses).mean()
		data['state_coord'] = s_next_coords

		# write summaries
		self.scalar_summaries['loss/{}/total'.format(mode)] += [loss.item()]
		self.list_summaries['loss/{}/total_histogram'.format(mode)] += torch.stack(losses).cpu().tolist()
		self.scalar_summaries['num_points/input'] += [(s.C[:, 0] == i).sum().item() for i in range(batch_size)]
		self.scalar_summaries['num_points/output'] += [one_hot_gt[i].shape[0] for i in range(batch_size)]
		self.list_summaries['scheduler/infusion_rates'] += infusion_rates

		if mode != 'train':
			return loss.detach().cpu().item(), data

		# take gradient descent
		self.zero_grad()
		loss.backward()
		self.clip_grad()
		self.optimizer.step()
		self.lr_scheduler.step()

		return loss.detach().cpu().item(), data

	def transition(self, s: SparseTensor, sigma=None) -> SparseTensor:
		y_hat = self.forward(s)
		feat_sample = self.sample_feat(y_hat.F)
		s_next_coord = y_hat.C[feat_sample.bool(), :]

		# if the sampled output contains no coords
		batch_size = s.C[:, 0].max().item() + 1
		for batch_idx in range(batch_size):
			if (s_next_coord[:, 0] == batch_idx).shape[0] == 0:
				if s_next_coord[:, 0].shape[0] == 0:
					s_next_coord = torch.zeros(1, 4).int().to(s_next_coord.device)
				else:
					s_next_coord = torch.stack([
						s_next_coord,
						torch.tensor([[batch_idx] + [0, ] * self.config['data_dim']]).int().to(s_next_coord.device)
					], dim=0)

		s_next_feat = torch.ones(s_next_coord.shape[0], 1)
		try:
			s_next = SparseTensor(
				s_next_feat, s_next_coord,
				device=self.device
			)
		except RuntimeError:
			breakpoint()
		return s_next

	def vis_collated_imgs(self, dataset, vis_indices: List, step: int):
		training = self.training
		self.eval()

		img_config = self.config['vis']['vis_collated_imgs']
		img_2d_config = img_config['vis_2d']
		img_3d_config = img_config['vis_3d']

		vis_batch_size = self.config['vis_batch_size']
		mini_batches = [
			vis_indices[i: i + vis_batch_size]
			for i in range(0, len(vis_indices), vis_batch_size)
		]

		for mini_batch_idxs in mini_batches:
			batch = [dataset[batch_idx] for batch_idx in mini_batch_idxs]
			batch_size = len(batch)
			batch = dataset.collate_fn(batch)

			state_coords = batch['state_coord']
			state_feats = batch['state_feat']
			embedding_coords = batch['embedding_coord']
			file_names = batch['file_name']
			num_phases = self.config['max_eval_phase']

			s = SparseTensor(
				features=torch.cat(state_feats),
				coordinates=ME.utils.batched_coordinates(state_coords),
				device=self.device,
			)

			if self.data_dim == 2:
				img_fn = vis_2d_coords
			elif self.data_dim == 3:
				img_fn = tensors2dist_func_tensor_imgs
			else:
				raise ValueError('data dim {} not allowed'.format(self.data_dim))

			# input
			input_coords = [
				s.C[s.C[:, 0] == batch_idx, 1:]
				for batch_idx in range(batch_size)
			]
			input_imgs = img_fn(input_coords, img_2d_config)
			input_3d_imgs = tensors2tensor_imgs(input_coords, self.data_dim, img_3d_config, batch_size)

			# ground truth
			gt_coords = [
				embedding_coords[batch_idx].detach().cpu()
				for batch_idx in range(batch_size)
			]
			gt_imgs = img_fn(gt_coords, img_2d_config)
			gt_3d_imgs = tensors2tensor_imgs(gt_coords, self.data_dim, img_3d_config, batch_size)

			output_imgs, output_3d_imgs, output_3d_imgs_batch = [], [], []
			phase = Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase']
			)

			for phase_cnt in range(num_phases):
				with torch.no_grad():
					s_next = self.transition(s)
				s = s_next
				phase += 1

				# transition output
				output_coords = [
					s.C[s.C[:, 0] == batch_idx, 1:]
					for batch_idx in range(batch_size)
				]
				output_imgs.append(img_fn(output_coords, img_2d_config))
				output_3d_imgs.append(
					tensors2tensor_imgs(output_coords, self.data_dim, img_3d_config, batch_size)
				)

			for batch_idx in range(batch_size):
				output_imgs_batch = torch.stack([
					output_imgs[phase][batch_idx]
					for phase in range(num_phases)
				], dim=0)
				output_3d_imgs_batch = torch.stack([
					output_3d_imgs[phase][batch_idx]
					for phase in range(num_phases)
				], dim=0)

				self.writer.add_video(
					'{}-img-{}'.format(dataset.mode, file_names[batch_idx]),
					torch.cat([
						input_imgs[batch_idx].unsqueeze(0).repeat_interleave(num_phases, dim=0),
						output_imgs_batch,
						gt_imgs[batch_idx].unsqueeze(0).repeat_interleave(num_phases, dim=0),
						input_3d_imgs[batch_idx].unsqueeze(0).repeat_interleave(num_phases, dim=0),
						output_3d_imgs_batch,
						gt_3d_imgs[batch_idx].unsqueeze(0).repeat_interleave(num_phases, dim=0),
					], dim=3).unsqueeze(0), global_step=step
				)

		self.train(training)

	def evaluate(self, data, step, dataset_mode) -> float:
		max_eval_phase = self.config['max_eval_phase']
		losses = []
		for mode in ['eval_infusion']:
			data_next = data
			for p in range(max_eval_phase):
				loss, data_next = self.learn(data_next, step, mode=mode)
				losses.append(loss)
		return sum(losses) / float(len(losses))

	def get_pointcloud(self, s: SparseTensor, sample_nums: List, return_mesh=True):
		ret = defaultdict(list)
		meshes = defaultdict(list)
		for batch_idx in range(s.C[:, 0].max().item() + 1):
			idx = s.C[:, 0] == batch_idx
			coord = s.C[idx, 1:]
			mesh = marching_cubes_sparse_voxel(coord, voxel_size=self.voxel_size)
			meshes['initial_mesh'] += [mesh]
			# if sample_num == 2048:
			# 	ret[sample_num] += [downsample(coord * self.voxel_size, sample_num)]
			# else:
			for sample_num in sample_nums:
				try:
					ret[sample_num] += [torch.tensor(mesh.sample(sample_num)).float()]
				except IndexError:
					ret[sample_num] += [torch.zeros(sample_num, 3)]  # for empty state
		if return_mesh:
			return ret, meshes
		return ret
