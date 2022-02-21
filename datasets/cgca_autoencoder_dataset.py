import torch
import numpy as np
import os
from datasets.base_dataset import BaseDataset
from models.base_model import Model
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.metrics import compute_chamfer_l1
from utils.util import quantize, downsample


class AutoencoderDataset(BaseDataset):
	def __init__(self, config: dict, mode: str):
		BaseDataset.__init__(self, config, mode)
		self.z_dim = config['z_dim']
		self.implicit_rep = config['implicit_rep']
		self.voxel_size = config['voxel_size']
		self.implicit_input_cnt = config['implicit_input_cnt']
		self.query_cnt = config['query_cnt']
		self.max_dist = config['max_dist']

	def convert_rep(self, signed_rep: torch.Tensor):
		"""
		:param signed_rep: torch.tensor of N
			Signed representation of the implicit field
		:return: rep: torch.tensor of N
			Converted representation
		"""
		if self.implicit_rep == 'sdf':
			return signed_rep
		elif self.implicit_rep == 'udf':
			return torch.abs(signed_rep)
		elif self.implicit_rep == 'occ':
			return (signed_rep > 0.).float()
		else:
			raise ValueError('representation {} not allowed'.format(self.implicit_rep))


class AutoencoderShapenetDataset(AutoencoderDataset):
	name = 'cgca_autoencoder_shapenet'

	def __init__(self, config: dict, mode: str):
		AutoencoderDataset.__init__(self, config, mode)
		self.obj_class = config['obj_class']
		self.summary_name = self.obj_class
		self.surface_cnt = config['surface_cnt']
		self.query_dist_filter = config['query_dist_filter_rate'] * self.max_dist

		if mode == 'train':
			self.data_root = os.path.join(
				config['data_root'], self.obj_class, 'train'
			)
			data_list_file_name = 'train.txt'
		elif mode == 'val' or mode == 'test':
			self.data_root = os.path.join(
				config['data_root'], self.obj_class, 'test'
			)
			data_list_file_name = 'test.txt'
		else:
			raise ValueError()

		data_list_file_path = os.path.join(
			config['data_root'], self.obj_class,
			data_list_file_name
		)
		with open(data_list_file_path, 'r') as f:
			self.data_list = f.read().splitlines()
		self.data_list = sorted([
			x[:-1] if x[-1] == '\n' else x
			for x in self.data_list
		])

		if (mode == 'val') and (config['eval_size'] is not None):
			# fix vis_indices
			eval_size = config['eval_size']
			if isinstance(eval_size, int):
				val_indices = torch.linspace(0, len(self.data_list) - 1, eval_size).int().tolist()
				self.data_list = [self.data_list[i] for i in val_indices]

	def __getitem__(self, idx):
		if self.config['overfit_one_ex'] is not None:
			idx = self.config['overfit_one_ex']

		data_name = self.data_list[idx]
		data_path = os.path.join(self.data_root, data_name + '.npz')
		data = np.load(data_path)

		surface = downsample(torch.tensor(data['surface']), self.surface_cnt)
		sdf_pos = data['sdf_pos']
		sdf_pos = torch.tensor(sdf_pos[~np.isnan(sdf_pos).any(axis=1)])
		sdf_neg = data['sdf_neg']
		sdf_neg = torch.tensor(sdf_neg[~np.isnan(sdf_neg).any(axis=1)])
		sdf = torch.cat([sdf_pos, sdf_neg], dim=0)
		sdf = sdf[torch.randperm(sdf.shape[0]), :]

		implicit_field = sdf[torch.abs(sdf[:, 3]) < self.voxel_size]
		implicit_field = downsample(implicit_field, self.implicit_input_cnt)

		query = sdf[torch.abs(sdf[:, 3]) < self.query_dist_filter]
		query = downsample(query, self.query_cnt)

		# translate
		if self.mode == 'train':
			translation = 4 * torch.rand([1, 4]) * self.voxel_size
			translation[0, 3] = 0.
		else:
			translation = torch.zeros([1, 4])
		surface = surface + translation[:, :3]
		query = query + translation
		implicit_field = implicit_field + translation

		# normalize
		surface = quantize(surface, self.voxel_size)
		query = query / self.voxel_size
		query_coord, query_val = query.split(3, 1)
		implicit_field = implicit_field / self.voxel_size
		query_val = query_val.view(-1)

		query_val = self.convert_rep(query_val)
		implicit_field[:, 3] = self.convert_rep(implicit_field[:, 3])

		return {
			'surface_voxel': surface,  # torch tensor of  N1 x 3
			'implicit_field': implicit_field,  # torch tensor of N2 x 4
			'query_coord': query_coord,  # torch tensor of N3 x 3
			'query_val': query_val,  # torch tensor of N3
			'translation': translation,  # torch tensor of 1 x 4
			'file_name': data_name,
			'path': data_path,
		}

	def __len__(self):
		return len(self.data_list)

	def test(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		# collect testset
		test_sample_num = self.config['test_sample_num']
		surfaces = {}
		for file_name in self.data_list:
			data_path = os.path.join(self.data_root, file_name + '.npz')
			data = np.load(data_path)
			surfaces[file_name] = torch.tensor(
				data['surface'][:test_sample_num]
			).float()

		print('Collected {} complete shapes'.format(len(surfaces)))

		data_loader = DataLoader(
			self,
			batch_size=self.config['test_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
			shuffle=False
		)

		test_chamfer_l1 = []
		for test_step, data in tqdm(enumerate(data_loader)):
			file_names = data['file_name']
			gts = [surfaces[file_name].to(self.device) for file_name in file_names]

			pred_pcs = model.get_pointcloud(data, step)

			for batch_idx, pred_pc in enumerate(pred_pcs):
				pred_coords_down = torch.stack(pred_pc, dim=0).to(self.device)
				chamfer_l1s = compute_chamfer_l1(pred_coords_down, gts[batch_idx])
				test_chamfer_l1.append(chamfer_l1s[0])

		chamfer_l1 = np.array(test_chamfer_l1).mean()
		print('chamfer_l1: {}'.format(chamfer_l1))

		# write to tensorboard
		model.scalar_summaries['metrics/chamfer_l1'] += [chamfer_l1]

		model.write_dict_summaries(step)
		model.train(training)
