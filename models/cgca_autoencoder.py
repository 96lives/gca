import MinkowskiEngine as ME
import os
import torch
import trimesh
from tqdm import tqdm
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from models.base_model import Model
from models.pointnet import PatchEncoder, PatchDecoder
from models.sparse_ifnet import SparseIFNet
from utils.marching_cube import marching_cube
from utils.util import timeit, downsample
from utils.solvers import build_lr_scheduler, build_optimizer
from utils.visualization import vis_2d_coords, tensors2dist_func_tensor_imgs
from utils.pad import get_shifts
from typing import Dict, List
from MinkowskiEngine import SparseTensor, MinkowskiInterpolationFunction


class CGCAAutoencoder(Model):
	name = 'cgca_autoencoder'

	def __init__(self, config: Dict, writer: SummaryWriter):
		Model.__init__(self, config, writer)
		self.implicit_rep = self.config['implicit_rep']
		self.encoder = PatchEncoder(config)
		self.conv_z = config['conv_z']
		if self.conv_z:
			self.decoder_conv = SparseIFNet(config)
		self.decoder_mlp = PatchDecoder(config)
		self.optimizer = build_optimizer(
			self.config['optimizer'], self.parameters()
		)
		self.lr_scheduler = build_lr_scheduler(
			self.config['lr_scheduler'], self.optimizer
		)
		if self.config.get('init_pretrained') is not None:
			checkpoint = torch.load(self.config['init_pretrained'])
			self.load_state_dict(checkpoint['model_state_dict'])

	def forward(self, implicit_fields, query_points):
		z = self.encoder_forward(implicit_fields)
		z_unfolded = self.decoder_conv_forward(z) if self.conv_z else z
		dist, query_idxs = self.decoder_mlp_forward(z_unfolded, query_points)
		return dist, query_idxs, z

	@timeit
	def encoder_forward(self, implicit_fields):
		return self.encoder.forward(implicit_fields)

	@timeit
	def decoder_conv_forward(self, z):
		return self.decoder_conv.forward(z)

	@timeit
	def decoder_mlp_forward(self, z_unfolded, query_points):
		return self.decoder_mlp.forward(z_unfolded, query_points)

	@timeit
	def learn(
			self, data: dict,
			step: int, mode: str = 'train'
	) -> (dict, float):
		"""
		:param data: dict containing key, value pairs of
			- surface_voxel (List of tensor of N1 x data_dim)
			- implicit_field (List of tensor of N2 x {data_dim + 1}, 1 for implicit feature)
			- query_coord (List of tensor of N3 x data_dim)
			- query_val (List of tensor of N3)
		:param step: training step
		:param mode: mode of training
		:return:
			- loss: float of the current step's lose
		"""
		raw_implicit_fields = data['implicit_field']
		surface_voxels = data['surface_voxel']
		implicit_fields = self.filter_implicit_field(surface_voxels, raw_implicit_fields)

		query_coords = data['query_coord']
		preds, query_idxs, z = self.forward(implicit_fields, query_coords)
		preds = self.implicit_activation(torch.cat(preds))

		query_vals = data['query_val']
		query_vals = torch.cat([
			dist[query_idx]
			for dist, query_idx in zip(query_vals, query_idxs)
		]).to(self.device)
		implicit_loss = self.implicit_loss(preds, query_vals)
		reg_loss = torch.norm(z.F, dim=1).mean()
		total_loss = implicit_loss + self.config['reg_loss_weight'] * reg_loss

		total_loss_scalar = total_loss.detach().cpu().item()
		self.scalar_summaries['loss/{}/implicit'.format(mode)] \
			+= [implicit_loss.detach().cpu().item()]
		self.scalar_summaries['loss/{}/reg'.format(mode)] \
			+= [reg_loss.detach().cpu().item()]
		self.scalar_summaries['loss/{}/total'.format(mode)] += [total_loss_scalar]
		self.scalar_summaries['loss/{}/implicit_rate'.format(mode)] \
			+= [implicit_loss.detach().cpu().item() / total_loss_scalar]

		if mode != 'train':
			return total_loss_scalar

		batch_size = len(implicit_fields)
		self.scalar_summaries['batch_size'] += [batch_size]
		self.scalar_summaries['points_cnt/raw_input_implicit_mean'] \
			+= [torch.cat(raw_implicit_fields, dim=0).shape[0] / batch_size]
		self.scalar_summaries['points_cnt/input_implicit_mean'] \
			+= [torch.cat(implicit_fields, dim=0).shape[0] / batch_size]
		self.list_summaries['points_cnt/input_implicit_max'] \
			+= [max([x.shape[0] for x in implicit_fields])]
		self.scalar_summaries['points_cnt/input_voxel_mean'] \
			+= [torch.cat(surface_voxels, dim=0).shape[0] / batch_size]
		self.list_summaries['points_cnt/input_voxel_max'] \
			+= [max([sv.shape[0] for sv in surface_voxels])]
		self.scalar_summaries['points_cnt/query_mean'] \
			+= [torch.cat(query_coords, dim=0).shape[0] / batch_size]
		self.list_summaries['points_cnt/query_max'] \
			+= [max([qc.shape[0] for qc in query_coords])]

		self.zero_grad()
		total_loss.backward()
		self.clip_grad()
		self.optimizer.step()
		self.lr_scheduler.step()
		return total_loss_scalar

	@timeit
	def filter_implicit_field(self, surface_voxels, implicit_fields, use_hash=False):
		"""
		Filters implicit surface to be contained in the voxel
		:param surface_voxels: List of tensor of N x 3 denoting voxel coordinates
			N is different for elements in the list
		:param implicit_fields: List of tensor of M x (data_dim + 1)
			M is different for elements in the list
		:param use_hash: boolean value
			if true, uses large boolean hash table to filter out the implcit values
		:return: filtered_implicit_field: where the implicit
			List of tensors
		"""
		if self.config.get('voxel_indices_root') is not None:
			return implicit_fields

		if use_hash:
			filtered_implicit_fields = []
			for surface_voxel, implicit_field in zip(surface_voxels, implicit_fields):
				# create hash table
				surface_voxel = surface_voxel.to(self.device)
				implicit_coord = (torch.round(implicit_field[:, :self.data_dim]).to(torch.int32)).to(self.device)
				all_coords = torch.cat([surface_voxel, implicit_coord], dim=0)
				bbox_min = all_coords.min(dim=0).values
				bbox_max = all_coords.max(dim=0).values
				bbox_diff = bbox_max - bbox_min
				hash_table = torch.zeros(torch.prod(bbox_diff).item() + 1, dtype=torch.bool)

				# set true for the occupied voxels
				offset = torch.tensor([torch.prod(bbox_diff[: dim]) for dim in range(self.data_dim)]).to(self.device)
				surface_voxel_hash_key = (surface_voxel - bbox_min.unsqueeze(0)) * offset.unsqueeze(0)
				hash_table[surface_voxel_hash_key] = True

				implicit_coord_hash_key = (implicit_coord - bbox_min.unsqueeze(0)) * offset.unsqueeze(0)
				implicit_coord_hash_value = hash_table[implicit_coord_hash_key]
				filtered_implicit_fields.append(implicit_field[implicit_coord_hash_value[:, 0], :])
			return filtered_implicit_fields

		feats = [
			torch.ones(sv.shape[0], 1)
			for sv in surface_voxels
		]
		sparse_surface = SparseTensor(
			features=torch.cat(feats),
			coordinates=ME.utils.batched_coordinates(surface_voxels),
			device=self.device
		)

		implicit_coord = ME.utils.batched_coordinates([
			torch.round(x[:, :self.data_dim])
			for x in implicit_fields
		]).to(self.device).float()
		# use unique to reduce the amount of implicit_coord called on interpolation function
		implicit_coord, inverse_idx = torch.unique(implicit_coord, dim=0, return_inverse=True)
		interpolation_feats = MinkowskiInterpolationFunction.apply(
			sparse_surface.F, implicit_coord,
			sparse_surface.coordinate_map_key,
			sparse_surface.coordinate_manager
		)[0]
		interpolation_feats = interpolation_feats[inverse_idx]
		interpolation_feats = interpolation_feats.view(-1).split([
			x.shape[0] for x in implicit_fields
		])
		return [
			implicit_fields[batch_idx][iv == 1]
			for batch_idx, iv in enumerate(interpolation_feats)
		]

	def get_query_points(self, voxels, grid_query=True):
		"""
		:param voxels: List of torch.coords or SparseTensor
		:param grid_query: boolean value for determining form of grid points
			if true, use grid_query (useful for marching cubes)
			else, obtain query points using gaussian noise around voxels
		:return:
			List of query points
		"""
		if type(voxels) == SparseTensor:
			voxels = [
				voxels.C[voxels.C[:, 0] == batch_idx, 1:]
				for batch_idx in range(voxels.C[:, 0].max() + 1)
			]
		query_coords = []
		for voxel in voxels:
			voxel = voxel.to(self.device)
			shifts = get_shifts(self.config['upsample'] - 1, self.data_dim).to(self.device)
			shifts = shifts.float() / self.config['upsample']
			query_coord = voxel
			# iterate to optimize memory footprint
			# otherwise use combination of torch.repeat() with torch.repeat_interleave()
			for shift in shifts:
				query_coord = torch.cat([
					query_coord,
					voxel + shift.view(1, -1)
				], dim=0)
				query_coord = torch.unique(query_coord, dim=0)
			query_coords.append(query_coord.cpu())

		return query_coords

	def encode(self, surface_voxels, implicit_fields):
		"""
		:param surface_voxels: List of torch of tensor N1 x self.data_dim
		:param implicit_fields: List of torch of tensor of N2 x self.data_dim
		:return z: SparseTensor containing encoded features
		"""
		implicit_fields = self.filter_implicit_field(surface_voxels, implicit_fields)
		return self.encoder_forward(implicit_fields)

	def autoencode(self, batch, query_near_voxel=False):
		"""
		:param batch: dictionary containing
			- surface_voxel: List of torch of tensor N1 x self.data_dim
			- implicit_field: List of torch of tensor of N2 x self.data_dim
			- query_coord: List of torch of tensor of N3 x self.data_dim
		:param query_near_voxel: boolean
			if true, generate new query points based on the voxel coordinates of z
			else, use the query points in the batch data.
			Note that this changes the query_coord in the batch
		:return:
		"""

		with torch.no_grad():
			implicit_fields = self.filter_implicit_field(batch['surface_voxel'], batch['implicit_field'])
			if query_near_voxel:
				query_coords = self.get_query_points(batch['surface_voxel'])
				batch['query_coord'] = query_coords
				batch['query_val'] = [None for _ in range(len(query_coords))]  # to explicitly identify batch changed

			preds, query_idxs, z = self.forward(implicit_fields, batch['query_coord'])
			preds = [self.implicit_activation(p) for p in preds]
		return preds, query_idxs, z

	def get_dists(self, z, query_points):
		self.decoder_mlp = self.decoder_mlp.to(self.device)
		dists, query_idxs = self.decoder_mlp.forward(z, query_points)
		query_points = [
			query_point[query_idx]
			for query_point, query_idx in zip(query_points, query_idxs)
		]
		dists = [self.implicit_activation(dist) for dist in dists]
		return dists, query_points

	def decode(self, z, grid_query=True):
		"""
		:param z: SparseTensor containing batched coordinates and features
		:param grid_query: boolean value for determining the query points
		:return:
			dists, query_points
		"""
		decoder_conv = self.decoder_conv.to(self.device)
		z_unfolded = decoder_conv.forward(z) if self.conv_z else z
		query_points = self.get_query_points(z, grid_query=grid_query)

		# divide query points if too many
		len_query = [qp.shape[0] for qp in query_points]
		max_query_cnt = max(len_query)

		# to fit in 11GB GPU for inference
		if max_query_cnt > 1000000:
			div_cnt = (max_query_cnt // 1000000) + 1
			query_points_div = list(zip(*[qp.chunk(div_cnt) for qp in query_points]))
			dists, query_points = [], []
			for query_point in query_points_div:
				dist, query_point = self.get_dists(z_unfolded, query_point)
				dists.append(dist)
				query_points.append(query_point)
			# transform div_cnt x batch_size -> batch_size x div_cnt and concat to batch_size
			dists = [torch.cat(dist) for dist in list(zip(*dists))]
			query_points = [torch.cat(qp) for qp in list(zip(*query_points))]
		else:
			dists, query_points = self.get_dists(z_unfolded, query_points)

		return dists, query_points

	def implicit_activation(self, dist):
		if self.implicit_rep == 'sdf':
			return torch.tanh(dist)
		elif (self.implicit_rep == 'udf') or (self.implicit_rep == 'occ'):
			return torch.sigmoid(dist)
		else:
			raise ValueError('implicit representation {} not allowed'.format(self.implicit_rep))

	def implicit_loss(self, pred, gt):
		if (self.implicit_rep == 'sdf') or (self.implicit_rep == 'udf'):
			return torch.abs(pred - torch.clamp(gt, min=-1., max=1.)).mean()
		elif self.implicit_rep == 'occ':
			return binary_cross_entropy_with_logits(pred, gt)
		else:
			raise ValueError('implicit representation {} not allowed'.format(self.implicit_rep))

	def evaluate(self, data: dict, step: int, mode: str):
		with torch.no_grad():
			loss = self.learn(data, step, mode)
		return loss

	def create_meshes(
			self,
			preds: List[torch.Tensor],
			pred_coords: List[torch.Tensor],
			z: SparseTensor
	) -> Dict[str, List[trimesh.Trimesh]]:
		"""
		:param preds: List of tensors of size N
		:param pred_coords: List of tensors of size N x 3 denoting coordinates:
		:return:

		"""
		img_config = self.config['vis']['vis_mesh']
		mesh_dict = defaultdict(list)

		for batch_idx in range(len(preds)):
			mesh_init = marching_cube(
				query_points=pred_coords[batch_idx].detach().cpu(),
				df=torch.abs(preds[batch_idx].detach().cpu()),
				march_th=img_config['march_th'],
				upsample=self.config['upsample'],
				voxel_size=self.config['voxel_size']
			)
			mesh_dict['initial_mesh'] += [mesh_init]

		return dict(mesh_dict)


	def vis_2d(self, dataset, vis_indices: List, step: int):
		training = self.training
		self.eval()

		img_config = self.config['vis']['vis_2d']
		vis_batch_size = self.config['vis_batch_size']
		mini_batches = [
			vis_indices[i: i + vis_batch_size]
			for i in range(0, len(vis_indices), vis_batch_size)
		]

		for mini_batch_idxs in mini_batches:
			batch = [dataset[batch_idx] for batch_idx in mini_batch_idxs]
			batch_size = len(batch)
			batch = dataset.collate_fn(batch)

			implicit_fields = batch['implicit_field']
			query_coords = batch['query_coord']
			query_vals = batch['query_val']
			file_names = batch['file_name']

			preds, query_idxs, z = self.autoencode(batch)

			if self.data_dim == 2:
				img_fn = vis_2d_coords
			elif self.data_dim == 3:
				img_fn = tensors2dist_func_tensor_imgs
			else:
				raise ValueError('data dim {} not allowed'.format(self.data_dim))

			# input
			input_coords = [x[:, :self.data_dim] for x in implicit_fields]
			input_feats = [torch.abs(x[:, self.data_dim]) for x in implicit_fields]
			input_imgs = img_fn(input_coords, img_config, input_feats)

			# voxel
			voxel_coords = [
				z.C[z.C[:, 0] == batch_idx, 1:]
				for batch_idx in range(batch_size)
			]
			voxel_imgs = img_fn(voxel_coords, img_config)

			# prediction
			pred_coords = [q[idx] for q, idx in zip(query_coords, query_idxs)]
			abs_preds = [torch.abs(p) for p in preds]
			pred_imgs = img_fn(pred_coords, img_config, abs_preds)

			# ground truth
			query_feats = [torch.abs(v) for v in query_vals]
			gt_imgs = img_fn(query_coords, img_config, query_feats)

			# difference btw prediction and ground truth
			diff = [
				torch.abs(p.cpu() - torch.clamp(q[idx], min=-1, max=1))
				for q, p, idx in zip(query_vals, preds, query_idxs)
			]
			diff_imgs = img_fn(pred_coords, img_config, diff)

			for batch_idx in range(batch_size):
				self.writer.add_image(
					'{}-img-{}'.format(dataset.mode, file_names[batch_idx]),
					torch.cat([
						input_imgs[batch_idx],
						voxel_imgs[batch_idx],
						pred_imgs[batch_idx],
						gt_imgs[batch_idx],
						diff_imgs[batch_idx]
					], dim=2), global_step=step
				)

		self.train(training)

	def vis_mesh(self, dataset, vis_indices: List, step: int):
		vis_batch_size = self.config['vis_batch_size']
		mini_batches = [
			vis_indices[i: i + vis_batch_size]
			for i in range(0, len(vis_indices), vis_batch_size)
		]

		for i, mini_batch_idxs in tqdm(enumerate(mini_batches)):
			batch = [dataset[i] for i in mini_batch_idxs]
			batch = dataset.collate_fn(batch)

			# prediction
			preds, query_idxs, z = self.autoencode(batch, query_near_voxel=True)
			pred_coords = [q[idx] for q, idx in zip(batch['query_coord'], query_idxs)]

			# create meshes
			mesh_dict = self.create_meshes(preds, pred_coords, z)

			# save meshes
			mesh_save_dir = os.path.join(
				self.config['log_dir'],
				'meshes', 'step-{}'.format(step + 1)
			)
			for k, meshes in mesh_dict.items():
				for batch_idx, mesh in enumerate(meshes):
					file_name = batch['file_name'][batch_idx]
					os.makedirs(os.path.join(mesh_save_dir, k), exist_ok=True)
					mesh.export(os.path.join(mesh_save_dir, k, '{}.obj'.format(file_name)))

	def get_pointcloud(self, data: Dict, step: float):
		training = self.training
		self.eval()

		preds, query_idxs, _ = self.autoencode(data, query_near_voxel=True)
		pred_coords = [q[idx] for q, idx in zip(data['query_coord'], query_idxs)]

		final_pcs = []
		dist_thres = self.config['test_dist_threshold']
		final_pcs.append([
			downsample(
				qp[torch.abs(pred) < dist_thres].cpu().float() * self.config['voxel_size'],
				self.config['test_sample_num']
			)
			for qp, pred in zip(pred_coords, preds)
		])
		self.train(training)
		return list(zip(*final_pcs))


