import torch
import MinkowskiEngine as ME
import yaml
import os
from models.transition_model import TransitionModel
from models.cgca_autoencoder import CGCAAutoencoder
from torch.utils.tensorboard import SummaryWriter
from utils.scheduler import InfusionScheduler, SigmaScheduler
from MinkowskiEngine import SparseTensor, MinkowskiInterpolationFunction
from utils.util import timeit, downsample
from utils.kl import bernoulli_kl, gaussian_kl
from utils.visualization import (
	vis_2d_coords, tensors2dist_func_tensor_imgs, tensors2tensor_imgs
)
from utils.phase import Phase
from typing import List
from collections import defaultdict


class CGCATransitionModel(TransitionModel):
	name = 'cgca_transition'

	def __init__(self, config, writer: SummaryWriter):
		self.z_dim = config['z_dim']
		config['backbone']['in_channels'] = self.z_dim
		config['backbone']['out_channels'] = self.z_dim + 1
		TransitionModel.__init__(self, config, writer)
		self.infusion_scheduler = InfusionScheduler(config)
		self.sigma_scheduler = SigmaScheduler(config)
		self.bce_loss = torch.nn.BCEWithLogitsLoss()
		self.data_dim = config['data_dim']
		self.pruning = ME.MinkowskiPruning()

	def get_out_coord(self, s: SparseTensor):
		shifts = self.shifts.repeat(s.shape[0], 1)
		s_repeat = torch.repeat_interleave(s.coordinates, self.shift_size, dim=0)
		return torch.unique(s_repeat + shifts, dim=0).int()

	@timeit
	def forward(self, s: SparseTensor) -> SparseTensor:
		"""
		Forward pass through sparse convolution network
		:param s: SparseTensor of
			coordinates with shape N x data_dim
			features with shape N x z_dim
		:return s_hat: SparseTensor of next output
		"""
		return self.backbone.forward(s, out_coord=self.get_out_coord(s))

	@timeit
	def transition(self, s: SparseTensor, sigma) -> SparseTensor:
		out = self.forward(s)
		if sigma is None:
			occ = out.F[:, 0] > 0.
			next_coord = out.C[occ]
			next_feat = out.F[occ, 1:]
		else:
			occ = torch.bernoulli(torch.sigmoid(out.F[:, 0])).bool()
			next_coord = out.C[occ]
			mu = out.F[occ, 1:]
			next_feat = mu + sigma * torch.randn_like(mu)

		return SparseTensor(
			features=next_feat,
			coordinates=next_coord,
			device=self.device
		)

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

		# prepare for forward pass
		s = SparseTensor(
			features=torch.cat(data['state_feat']),
			coordinates=ME.utils.batched_coordinates(data['state_coord']),
			device=self.device,
		)

		# forward pass
		out = self.forward(s)

		phases = data['phase']
		batch_size = len(phases)
		infusion_rates = self.infusion_scheduler.sample(phases, mode=mode)
		sigmas = self.sigma_scheduler.sample(phases)

		voxel_losses = []
		unfolded_infusion_rates = []
		unfolded_sigmas = []
		q_occs = []
		out_idxs = []

		for batch_idx, (infusion_rate, sigma, phase) in \
				enumerate(zip(infusion_rates, sigmas, phases)):
			out_idx = out.C[:, 0] == batch_idx
			out_idxs.append(out_idx)
			out_coord = out.C[out_idx, 1:]
			embedding_coord = data['embedding_coord'][batch_idx].to(self.device)

			# compute voxel loss
			with torch.no_grad():
				chamfer_dist, _, _, infusion_idx = self._chamfer_dist(
					out_coord.unsqueeze(0).float().to(self.device),
					embedding_coord.unsqueeze(0).float().to(self.device),
					return_idx=True
				)

			infusion_idx = torch.unique(infusion_idx.squeeze(0)).long()
			infusion_feat = torch.zeros(out_coord.shape[0]).to(self.device)
			infusion_feat[infusion_idx] = 1.

			p_occ = torch.sigmoid(out.F[out_idx, 0])
			q_occ = infusion_rate * infusion_feat + (1. - infusion_rate) * p_occ
			voxel_loss = bernoulli_kl(q_occ, p_occ)
			voxel_losses.append(voxel_loss)

			unfolded_infusion_rates.append(infusion_rate * torch.ones(out_idx.sum()).to(self.device))
			unfolded_sigmas.append(sigma * torch.ones(out_idx.sum()).to(self.device))
			q_occs.append(q_occ)

			# update_phases
			phases[batch_idx] = phase + 1
			completion_rate = (chamfer_dist == 0).sum().item() / float(embedding_coord.shape[0])
			if completion_rate >= self.config['completion_rate']:
				if not phase.equilibrium_mode:
					phase.set_complete()
					self.list_summaries['completion_phase/{}'.format(mode)] += [phase.phase]
			elif (phase.phase > self.config['max_phase']) and (mode == 'train'):
				incomplete_key = 'phase/incomplete_cnt'
				self.scalar_summaries[incomplete_key] = [self.scalar_summaries[incomplete_key][0] + 1] if \
					len(self.scalar_summaries[incomplete_key]) != 0 else [1]

		# compute embedding loss
		sparse_embedding = SparseTensor(
			features=torch.cat(data['embedding_feat'], dim=0),
			coordinates=ME.utils.batched_coordinates(data['embedding_coord']),
			device=self.device
		)

		stacked_infusion_rate = torch.cat(unfolded_infusion_rates).view(-1, 1)
		mu_p = out.F[:, 1:]
		z = self.interpolate(sparse_embedding, out.C.float())

		# this bug occurs to me.interpolate if every z is zero
		if z.shape != mu_p.shape:
			z = z.view(*mu_p.shape)
		mu_q = ((1 - stacked_infusion_rate) * mu_p) + (stacked_infusion_rate * z)

		stacked_embedding_loss = torch.cat(q_occs) \
		                         * gaussian_kl(mu_q, mu_p, torch.cat(unfolded_sigmas))

		voxel_loss = torch.cat(voxel_losses).mean()
		embedding_loss = stacked_embedding_loss.mean()
		total_loss = voxel_loss + (self.config['embedding_loss_weight'] * embedding_loss)

		# write summaries
		self.scalar_summaries['loss/{}/total'.format(mode)] \
			+= [total_loss.detach().cpu().item()]
		self.scalar_summaries['loss/{}/voxel'.format(mode)] \
			+= [voxel_loss.detach().cpu().item()]
		self.scalar_summaries['loss/{}/embedding'.format(mode)] \
			+= [embedding_loss.detach().cpu().item()]
		self.scalar_summaries['loss/{}/embedding'.format(mode)] \
			+= [embedding_loss.detach().cpu().item()]
		self.scalar_summaries['num_points/{}/input'.format(mode)] \
			+= [s.shape[0] / float(batch_size)]
		self.scalar_summaries['num_points/{}/neighborhood'.format(mode)] \
			+= [out.shape[0] / float(batch_size)]
		self.scalar_summaries['resources/batch_size'] += [len(phases)]

		# update next step
		s_next_coords = []
		s_next_feats = []
		for batch_idx in range(batch_size):
			# get coordinates
			occ_idx = torch.bernoulli(q_occs[batch_idx]).bool()
			s_next_coords.append(out.C[out_idxs[batch_idx], 1:][occ_idx].detach().cpu())

			# get features
			next_mu_q = mu_q[out_idxs[batch_idx]][occ_idx]
			next_feat = next_mu_q + (sigmas[batch_idx] * torch.randn_like(next_mu_q))
			s_next_feats.append(next_feat.detach().cpu())

		data['state_coord'] = s_next_coords
		data['state_feat'] = s_next_feats

		if mode != 'train':
			return total_loss.detach().cpu().item(), data

		# take gradient descent
		self.zero_grad()
		total_loss.backward()
		self.clip_grad()
		self.optimizer.step()
		self.lr_scheduler.step()

		return total_loss.detach().cpu().item(), data

	@timeit
	def interpolate(self, s: SparseTensor, query: torch.Tensor):
		return MinkowskiInterpolationFunction.apply(
			s.F, query,
			s.coordinate_map_key,
			s.coordinate_manager
		)[0]

	def evaluate(self, data, step, dataset_mode) -> float:
		max_eval_phase = self.config['max_eval_phase']
		losses = []
		for mode in ['eval_infusion']:
			data_next = data
			for p in range(max_eval_phase):
				with torch.no_grad():
					loss, data_next = self.learn(data_next, step, mode=mode)
				losses.append(loss)
		return sum(losses) / float(len(losses))

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
			embedding_feats = batch['embedding_feat']
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
			diff_imgs, diff_tenfold_imgs = [], []
			phase = Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase']
			)

			if self.config['model'] == 'cgca_transition_condition':
				self.register_s0(batch['input_pc'])

			for phase_cnt in range(num_phases):
				with torch.no_grad():
					s_next = self.transition(s, sigma=self.sigma_scheduler.sample([phase])[0])
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
				query = [
					torch.cat([
						batch_idx * torch.ones(ec.shape[0], 1),
						ec.float()
					], dim=1).to(self.device)
					for batch_idx, ec in enumerate(embedding_coords)
				]

				diff, diff_tenfold = [], []
				for batch_idx in range(batch_size):
					latents, _, _, _ = MinkowskiInterpolationFunction.apply(
						s.F, query[batch_idx], s.coordinate_map_key, s.coordinate_manager
					)
					# difference btw prediction and ground truth
					gt_feats = embedding_feats[batch_idx]
					preds = torch.norm(gt_feats - latents.cpu(), dim=1)
					diff.append(preds)
					diff_tenfold.append(10 * preds)

				diff_imgs.append(img_fn(gt_coords, img_2d_config, diff))
				diff_tenfold_imgs.append(img_fn(gt_coords, img_2d_config, diff_tenfold))

			for batch_idx in range(batch_size):
				output_imgs_batch = torch.stack([
					output_imgs[phase][batch_idx]
					for phase in range(num_phases)
				], dim=0)
				output_3d_imgs_batch = torch.stack([
					output_3d_imgs[phase][batch_idx]
					for phase in range(num_phases)
				], dim=0)
				diff_imgs_batch = torch.stack([
					diff_imgs[phase][batch_idx]
					for phase in range(num_phases)
				], dim=0)
				diff_tenfold_batch = torch.stack([
					diff_tenfold_imgs[phase][batch_idx]
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
						diff_imgs_batch,
						diff_tenfold_batch,
					], dim=3).unsqueeze(0), global_step=step
				)

		self.train(training)

	def load_ae(self):
		embedding_dir = self.config['embedding_root']
		ae_config = yaml.load(
			open(os.path.join(embedding_dir, 'ae_config.yaml')),
			Loader=yaml.FullLoader
		)
		self.ae = CGCAAutoencoder(ae_config, self.writer)
		checkpoint = torch.load(os.path.join(embedding_dir, 'ae.pt'))
		self.ae.load_state_dict(checkpoint['model_state_dict'])

	def get_pointcloud(self, s: SparseTensor, sample_nums: List, return_mesh=True):
		if not hasattr(self, 'ae'):
			self.load_ae()

		with torch.no_grad():
			preds, query_points = self.ae.decode(s, grid_query=True)
		self.ae.config['vis']['vis_mesh']['simplify'] = False
		meshes = self.ae.create_meshes(preds, query_points, s)

		ret = defaultdict(list)
		dist_thres = self.config['test_dist_threshold']
		for sample_num in sample_nums:
			for qp, pred in zip(query_points, preds):
				qp = qp[torch.abs(pred) < dist_thres].cpu().float() * self.config['voxel_size']
				ret[sample_num] = [downsample(qp, sample_num)]
		torch.cuda.empty_cache()
		if return_mesh:
			return ret, meshes
		return ret

	def create_meshes(self, s: SparseTensor):
		if not getattr(self, 'ae'):
			self.load_ae()
		with torch.no_grad():
			preds, query_points = self.ae.decode(s, grid_query=True)
		torch.cuda.empty_cache()
		return self.ae.create_meshes(preds, query_points, s)


class CGCATransitionConditionModel(CGCATransitionModel):
	name = 'cgca_transition_condition'

	def __init__(self, config, writer: SummaryWriter):
		self.z_dim = config['z_dim']
		config['backbone']['in_channels'] = 2 * self.z_dim
		config['backbone']['out_channels'] = self.z_dim + 1
		TransitionModel.__init__(self, config, writer)
		self.infusion_scheduler = InfusionScheduler(config)
		self.sigma_scheduler = SigmaScheduler(config)
		self.bce_loss = torch.nn.BCEWithLogitsLoss()
		self.data_dim = config['data_dim']
		self.pruning = ME.MinkowskiPruning()
		self.union = ME.MinkowskiUnion()

		# load autoencoder
		embedding_dir = self.config['embedding_root']
		ae_config = yaml.load(
			open(os.path.join(embedding_dir, 'ae_config.yaml')),
			Loader=yaml.FullLoader
		)
		self.ae = CGCAAutoencoder(ae_config, writer)
		checkpoint = torch.load(os.path.join(embedding_dir, 'ae.pt'))
		self.ae.load_state_dict(checkpoint['model_state_dict'])

		# disable autoencoder training
		for param in self.ae.parameters():
			param.requires_grad = False

		self.s0 = None

	@timeit
	def learn(
			self, data: dict,
			step: float, mode: str = 'train'
	) -> (dict, float):
		"""
		:param data: dict containing key, value pairs of
			- input_pc: Tensor containing coordinates of raw point cloud input
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

		# obtain state0 feat for initial states
		init_implicit_fields, init_idxs = [], []
		for batch_idx, input_pc in enumerate(data['input_pc']):
			if data['state0_feat'][batch_idx] is not None:
				continue
			init_implicit_fields.append(
				torch.cat([
					input_pc, torch.zeros(input_pc.shape[0], 1)
				], dim=1).to(self.device)
			)  # N x {data_dim + 1}
			init_idxs.append(batch_idx)

		if len(init_implicit_fields) != 0:
			with torch.no_grad():
				z = self.ae.encoder_forward(init_implicit_fields)

			init_cnt = 0
			for batch_idx, state0_feat in enumerate(data['state0_feat']):
				if state0_feat is not None:
					continue
				idx = z.C[:, 0] == init_cnt
				data['state0_coord'][batch_idx] = z.C[idx, 1:].cpu()
				data['state0_feat'][batch_idx] = z.F[idx].cpu()
				init_cnt += 1

		s0_feat = torch.cat(data['state0_feat'], dim=0)
		s0_feat = torch.cat([torch.zeros_like(s0_feat), s0_feat], dim=1)
		s0 = SparseTensor(
			features=s0_feat,
			coordinates=ME.utils.batched_coordinates(data['state0_coord']),
			device=self.device
		)
		s_feat = torch.cat(data['state_feat'], dim=0)
		s_feat = torch.cat([s_feat, torch.zeros_like(s_feat)], dim=1)
		s = SparseTensor(
			features=s_feat,
			coordinates=ME.utils.batched_coordinates(data['state_coord']),
			device=self.device,
			coordinate_manager=s0.coordinate_manager,
		)

		# override state_coord and state_feat using MinkUnion
		s_new = self.union(s0, s)
		for batch_idx in range(len(data['state0_feat'])):
			idx = s_new.C[:, 0] == batch_idx
			data['state_coord'][batch_idx] = s_new.C[idx, 1:].cpu()
			data['state_feat'][batch_idx] = s_new.F[idx].cpu()
		return CGCATransitionModel.learn(self, data, step, mode)

	def register_s0(self, input_pcs):
		"""
		:param input_pcs: List of torch.tensor
		:return:
		"""
		implicit_fields = [
			torch.cat([
				input_pc, torch.zeros(input_pc.shape[0], 1)
			], dim=1).to(self.device)
			for input_pc in input_pcs
		]
		with torch.no_grad():
			self.s0 = self.ae.encoder_forward(implicit_fields)
		self.s0._F = torch.cat([torch.zeros_like(self.s0.F), self.s0.F], dim=1)

	@timeit
	def transition(self, s: SparseTensor, sigma) -> SparseTensor:
		if self.s0 is None:
			raise RuntimeError('initial state s0 is not set')
		# must reset self.s0 due to mink union bug
		self.s0 = SparseTensor(
			features=self.s0.F,
			coordinates=self.s0.C,
			device=self.device,
		)
		s = SparseTensor(
			features=torch.cat([s.F, torch.zeros_like(s.F)], dim=1),
			coordinates=s.C,
			device=self.device,
			coordinate_manager=self.s0.coordinate_manager,
		)
		s_new = self.union(self.s0, s)
		s_new = SparseTensor(
			features=s_new.F,
			coordinates=s_new.C,
			device=self.device,
		)

		return CGCATransitionModel.transition(self, s_new, sigma)

