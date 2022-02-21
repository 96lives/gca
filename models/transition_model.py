import torch
import os
import MinkowskiEngine as ME
from torch.utils.tensorboard import SummaryWriter
from abc import ABC
from models.unet_backbones import BACKBONES
from models.base_model import Model
from typing import List
from utils.solvers import build_lr_scheduler, build_optimizer
from utils.pad import get_shifts
from utils.chamfer_distance import ChamferDistance
from MinkowskiEngine import SparseTensor
from glob import glob

"""
An abstract model for GCA and cGCA
"""

class TransitionModel(Model, ABC):
	def __init__(self, config, writer: SummaryWriter):
		Model.__init__(self, config, writer)
		# The output sparse tensor is packed with neighboring information
		self.backbone = BACKBONES[config['backbone']['name']](config)
		# initialize the sparse tensor to choose

		self.out_dim = config['backbone']['out_channels']
		self.optimizer = build_optimizer(
			self.config['optimizer'], self.parameters()
		)
		self.lr_scheduler = build_lr_scheduler(
			self.config['lr_scheduler'], self.optimizer
		)
		self.shifts = get_shifts(
			padding=self.config['padding'],
			pad_type=self.config['pad_type'],
			data_dim=self.config['data_dim'],
			include_batch=True
		).to(self.config['device'])
		self.shift_size = self.shifts.shape[0]
		self._chamfer_dist = ChamferDistance()
		self.sampling_scheme = self.config['sampling_scheme']
		self.voxel_size = self.config['voxel_size']

	def transition(self, *args, **kwargs) -> SparseTensor:
		raise NotImplementedError()

	def sample_feat(self, feat: torch.tensor, sampling_scheme=None) -> torch.tensor:
		'''
		Takes feature of torch tensor as input and returns sampled features
		Args:
			feat: torch.tensor of shape N x param_dim
			sampling_scheme: str of sampling scheme
		Output:
			tensor of shape N
		'''
		sampling_scheme = self.sampling_scheme \
			if sampling_scheme is None else sampling_scheme
		if sampling_scheme == 'bernoulli':
			assert feat.shape[1] == 1, \
				'Expected feature shape of 1 for bernoulli, but got {}'.format(feat.shape[1])
			return torch.bernoulli(torch.sigmoid(feat.squeeze(1)))
		elif sampling_scheme == 'ml':
			assert feat.shape[1] == 1, \
				'Expected feature shape of 1 for bernoulli, but got {}'.format(feat.shape[1])
			return (torch.sigmoid(feat.squeeze(1)) > 0.5).float()
		else:
			raise ValueError('sampling scheme {} not allowed'.format(sampling_scheme))

	def evaluate(self, x: SparseTensor, y: SparseTensor, step) -> float:
		raise NotImplementedError()

	def sparsetensors2cache_dicts(self, s: SparseTensor):
		'''
		:param s:
		:return:
		list of dicts ready to be cached
		'''
		batch_size = s.C[:, 0].max() + 1
		datas = []
		for batch_idx in range(batch_size):
			data = {}
			idx = s.C[:, 0] == batch_idx
			data['s_coord'] = s.C[idx, 1:].cpu()
			data['s_feat'] = s.F[idx, :].detach().cpu()
			datas.append(data)
		return datas

	def load_cache(self, step, file_names):
		'''
		:param step:
		:param file_names:
		:return:
			list of list of coord and feat tensors as dictionary
				where first list iterates through trials and second iterates through loaded files
		'''
		step_dir = os.path.join(
			self.config['log_dir'],
			'test_save', 'step-{}'.format(step),
		)
		cache_dir = os.path.join(step_dir, 'cache')

		# for backwards compatibility
		if not os.path.exists(cache_dir):
			cache_dir = os.path.join(step_dir, 'latent')

		data_dicts = []
		for file_name in file_names:
			file_paths = sorted(glob(os.path.join(cache_dir, file_name + '*.pt')))
			if len(file_paths) == 0:
				raise FileNotFoundError(
					'no file with name {} are found at {}'.format(file_name, cache_dir)
				)
			data_dicts_single_file = []
			for file_path in file_paths:
				data_dicts_single_file.append(torch.load(file_path))
			data_dicts.append(data_dicts_single_file)
		data_dicts = list(zip(*data_dicts))  # file x batch
		return data_dicts

	def cache_dicts2sparse_tensor(self, data_dicts: List[dict]):
		'''
		Decode cache dicts to sparse tensors
		:param data_dicts:
		:return:
			sparse tensors containing batched coordinates and feats
		'''
		coords, feats = [], []
		for data_dict in data_dicts:
			coord, feat = data_dict['s_coord'], data_dict['s_feat']
			if coord.shape[0] == 0:
				coord, feat = torch.zeros(1, 3), torch.randn(1, self.config['z_dim'])
			coords.append(coord)
			feats.append(feat)

		return SparseTensor(
			features=torch.cat(feats, dim=0),
			coordinates=ME.utils.batched_coordinates(coords),
			device=self.device
		)

	def get_pointcloud(self, *args, **kwargs):
		raise NotImplementedError()

	def create_meshes(self, *args, **kwargs):
		raise NotImplementedError()

