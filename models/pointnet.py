'''
A lot of the pointnet encoder-decoder has been originated from
Songyou Pengs' convolutional occupancy networks: https://github.com/autonomousvision/convolutional_occupancy_networks
'''
import torch
import torch.nn as nn
import MinkowskiEngine as ME
import torch.nn.functional as F
from MinkowskiEngine import SparseTensor, SparseTensorQuantizationMode, TensorField
from MinkowskiEngine.MinkowskiInterpolation import MinkowskiInterpolationFunction
from typing import List


class PatchPointNet(nn.Module):

	def __init__(self, config):
		nn.Module.__init__(self)
		self.config = config
		self.device = config['device']


class PatchEncoder(PatchPointNet):
	"""
	PointNet-based encoder network with ResNet blocks.
	First transform input points to local system based on the given voxel size.
	Support non-fixed number of point cloud, but need to precompute the index
	"""

	def __init__(self, config):
		PatchPointNet.__init__(self, config)

		# init all the hyperparameters
		self.z_dim = config['z_dim']
		self.voxel_size = config['voxel_size']
		self.hidden_dim = config['encoder']['hidden_dim']
		self.num_blocks = config['encoder']['num_blocks']
		self.data_dim = config['data_dim']
		self.input_dim = self.data_dim + 1

		# init network architecture
		self.fc_pos = nn.Linear(self.input_dim, 2 * self.hidden_dim)
		self.blocks = nn.ModuleList([
			ResnetBlockFC(2 * self.hidden_dim, self.hidden_dim)
			for _ in range(self.num_blocks)
		])

		self.fc_z = nn.Linear(self.hidden_dim, self.z_dim)
		self.relu = nn.ReLU()

	def forward(self, implicit_fields):
		"""
		:param
			- implicit_fields (list of tensor of N x {self.data_dim + 1}): input point cloud
			- sparse_output (boolean): if true, outputs sparse tensor, else dense voxel grid
		:return:
			- out (SparseTensor): voxel feature
		"""

		# map to local coords and indices
		vox_coords = [x[:, :self.data_dim] for x in implicit_fields]
		local_coords = [vc - torch.round(vc) for vc in vox_coords]
		vox_coords = ME.utils.batched_coordinates([
			torch.round(vc) for vc in vox_coords
		]).to(self.device)
		local_implicit_fields = torch.cat([
			torch.cat(local_coords, dim=0),
			torch.cat(implicit_fields)[:, self.data_dim].unsqueeze(1)
		], dim=1).to(self.device)  # M x {self.data_dim + 1}

		out = self.fc_pos(local_implicit_fields)
		out = self.blocks[0](out)

		tensor_field, sparse_tensor = None, None
		for block in self.blocks[1:]:
			pooled, tensor_field, sparse_tensor = self.pool_local_mink_tensorfield(out, vox_coords, tensor_field, sparse_tensor)
			out = torch.cat([out, pooled], dim=1)
			out = block(out)
		out = self.fc_z(out)

		_, out_tensor_field, _ = self.pool_local_mink_tensorfield(out, vox_coords, tensor_field, sparse_tensor)
		return out_tensor_field.sparse()

		# print('encoder shape: {}'.format(out.shape[0]))
		# return SparseTensor(
		# 	features=out, coordinates=vox_coords,
		# 	quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
		# 	device=self.device,
		# )

	def pool_local_mink(self, feat, vox_coords):
		'''
		:param feat (tensor N x {feat_dim})
		:param vox_coords (tensor of N x {data_dim + 1}): batched coordinates
		:return: avg_feat (torch of tensor N x {feat_dim}): features are averaged that are within the same voxel
		'''
		# we use ME's sparse tensor to efficiently average the sparse coordinates
		# note that this does not compute all the volumes of the 3D space resulting in sparse representation
		out = SparseTensor(
			features=feat,
			coordinates=vox_coords,
			quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
			device=self.device
		)
		return out.F[out.inverse_mapping]

	def pool_local_mink_tensorfield(self, feat, vox_coords, tensor_field=None, sparse_tensor=None):
		'''
		:param feat (tensor N x {feat_dim})
		:param vox_coords (tensor of N x {data_dim + 1}): batched coordinates
		:return: avg_feat (torch of tensor N x {feat_dim}): features are averaged that are within the same voxel
		'''
		# we use ME's sparse tensor to efficiently average the sparse coordinates
		# note that this does not compute all the volumes of the 3D space resulting in sparse representation
		if tensor_field is None:
			tensor_field = TensorField(
				features=feat,
				coordinates=vox_coords,
				quantization_mode=SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
				minkowski_algorithm=ME.MinkowskiAlgorithm.MEMORY_EFFICIENT,
				device=self.device,
			)
			sparse_tensor = tensor_field.sparse()
		else:
			tensor_field._F = feat
		return sparse_tensor.slice(tensor_field).F, tensor_field, sparse_tensor


class PatchDecoder(PatchPointNet):
	"""
	PointNet-based encoder network with ResNet blocks.
	First transform input points to local system based on the given voxel size.
	Support non-fixed number of point cloud, but need to precompute the index
	"""

	def __init__(self, config):
		PatchPointNet.__init__(self, config)

		# init all the hyperparameters
		self.voxel_size = config['voxel_size']
		self.data_dim = config['data_dim']
		self.z_dim = config['z_dim']

		self.num_blocks = config['decoder']['num_blocks']
		self.hidden_dim = config['decoder']['hidden_dim']
		self.leaky_activation = config['decoder']['leaky_activation']

		# init network architecture
		self.fc_p = nn.Linear(self.z_dim, self.hidden_dim)
		self.fc_c = nn.ModuleList([
			nn.Linear(self.z_dim, self.hidden_dim)
			for _ in range(self.num_blocks)
		])

		self.blocks = nn.ModuleList([
			ResnetBlockFC(self.hidden_dim)
			for _ in range(self.num_blocks)
		])
		self.fc_out = nn.Linear(self.hidden_dim, 1)

		if self.leaky_activation:
			self.actvn = lambda x: F.leaky_relu(x, 0.2)
		else:
			self.actvn = F.relu
		self.eps = 1e-3

	def forward(
			self, z: SparseTensor,
			query_points: List[torch.Tensor],
			ignore_first_dim=False,
			diff_interpolation=False
	) -> (List[torch.Tensor], List[torch.Tensor]):
		"""
		:param z: SparseTensor
			SparseTensor containing occupied coordinates and corresponding patch latents
		:param query_points: list of torch.Tensor
			list of tensors containing coordinates of query point
		:param ignore_first_dim: boolean
			ignores the first dimension of sparsetensor features
		:param diff_interpolation: boolean
			whether to use differentiable interpolation (w.r.t. query points)
		:return dist: list of torch.Tensor
			list of tensors containing distances for query points
		:return query_idxs: list of torch.Tensor
			list of tensors containing indices of tensors for each batch
		"""

		# obtain unbatched queries
		batch_size = len(query_points)
		unbatched_queries = ME.utils.batched_coordinates([
			query_points[i]
			for i in range(batch_size)
		], dtype=torch.float32).to(self.device)

		# obtain latents, when no latents exist, interpolate with zero latents

		if isinstance(z, tuple) or isinstance(z, list):
			latents = torch.zeros(unbatched_queries.shape[0], self.z_dim).to(self.device)
			out_idx = []
			for z_single in z:
				latent, out_idx_single = self.interpolate(z_single, unbatched_queries, diff_interpolation)
				latents = latents + latent
				out_idx.append(out_idx_single)
			out_idx = torch.cat(out_idx)
			out_idx = torch.unique(out_idx, sorted=True).long()
		else:
			z_new = SparseTensor(z.F, z.C)
			latents, out_idx = self.interpolate(z_new, unbatched_queries, diff_interpolation)

		latents = latents[out_idx, :]
		if ignore_first_dim:
			latents = latents[:, 1:]

		# forward pass using network
		net = self.fc_p(latents)
		for i in range(self.num_blocks):
			net = net + self.fc_c[i](latents)
			net = self.blocks[i](net)
		out = self.fc_out(self.actvn(net))
		out = out.squeeze(1)

		# obtain batch_mask
		query_idx_offset = torch.cumsum(
			torch.tensor([
				pts.shape[0] for pts in query_points
			]).to(self.device),
			dim=0
		)
		batch_mask = out_idx.unsqueeze(1) < query_idx_offset.unsqueeze(0)
		batch_mask = batch_size - batch_mask.float().sum(dim=1)

		# convert into list form
		query_idxs = [
			(out_idx[batch_mask == i] - query_idx_offset[i - 1]).long() if i != 0 \
			else out_idx[batch_mask == i].long()
			for i in range(batch_size)
		]
		dists = [
			out[batch_mask == i]
			for i in range(batch_size)
		]

		return dists, query_idxs

	def interpolate(self, z, queries, diff_interpolation):
		if diff_interpolation:
			latent, out_idx = self.differentiable_interpolation(z, queries)
		else:
			latent, _, out_map, _ = MinkowskiInterpolationFunction.apply(
				z.F, queries,
				z.coordinate_map_key,
				z.coordinate_manager
			)
			out_idx = torch.unique(out_map, sorted=True).long()
		return latent, out_idx

	def differentiable_interpolation(self, s: SparseTensor, query: torch.Tensor):
		if query.shape[1] == 3:
			offsets = torch.tensor([
				[0, 0], [0, 1],
				[1, 0], [1, 1]
			]).to(self.device)
		elif query.shape[1] == 4:
			offsets = torch.tensor([
				[0, 0, 0], [0, 0, 1],
				[0, 1, 0], [0, 1, 1],
				[1, 0, 0], [1, 0, 1],
				[1, 1, 0], [1, 1, 1],
			]).to(self.device)
		else:
			raise ValueError()

		# add batch dimension
		offsets = torch.cat([
			torch.zeros(offsets.shape[0], 1).to(self.device),
			offsets
		], dim=1)
		query_floor = torch.floor(query)
		query_rel = query - query_floor

		latent = torch.zeros(query.shape[0], s.F.shape[1]).to(self.device)
		out_idx = torch.zeros(query.shape[0]).to(self.device)
		for offset in offsets:
			offset = offset.view(1, -1)
			corner = query_floor + offset
			corner_latent, in_map, out_map, weights = MinkowskiInterpolationFunction.apply(
				s.F, corner,
				s.coordinate_map_key,
				s.coordinate_manager
			)
			multi_const = 2 * offset[:, 1:] - 1
			add_const = 1 - offset[:, 1:]
			weight = torch.prod(add_const + multi_const * query_rel[:, 1:], dim=1).view(-1, 1)
			latent += weight * corner_latent
			out_idx[out_map.long()] = 1
		return latent, out_idx.nonzero(as_tuple=True)[0]


# Resnet Blocks
class ResnetBlockFC(nn.Module):
	''' Fully connected ResNet Block class.

	Args:
		size_in (int): input dimension
		size_out (int): output dimension
		size_h (int): hidden dimension
	'''

	def __init__(self, size_in, size_out=None, size_h=None):
		super().__init__()
		# Attributes
		if size_out is None:
			size_out = size_in

		if size_h is None:
			size_h = min(size_in, size_out)

		self.size_in = size_in
		self.size_h = size_h
		self.size_out = size_out
		# Submodules
		self.fc_0 = nn.Linear(size_in, size_h)
		self.fc_1 = nn.Linear(size_h, size_out)
		self.actvn = nn.ReLU()

		if size_in == size_out:
			self.shortcut = None
		else:
			self.shortcut = nn.Linear(size_in, size_out, bias=False)
		# Initialization
		nn.init.zeros_(self.fc_1.weight)

	def forward(self, x):
		net = self.fc_0(self.actvn(x))
		dx = self.fc_1(self.actvn(net))

		if self.shortcut is not None:
			x_s = self.shortcut(x)
		else:
			x_s = x

		return x_s + dx
