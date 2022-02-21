import torch
import os
import random
import numpy as np
import MinkowskiEngine as ME
from glob import glob
from tqdm import tqdm
from datasets.base_dataset import BaseDataset
from collections import defaultdict
from models.transition_model import TransitionModel
from MinkowskiEngine import SparseTensor, MinkowskiInterpolationFunction
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.util import downsample, quantize
from utils.phase import Phase
from utils.metrics import (
	mutual_difference,
	unidirected_hausdorff_distance,
	MMDCalculator,
	compute_chamfer_l1
)
from utils.visualization import (
	sparse_tensors2tensor_imgs, save_tensor_img, tensors2tensor_imgs
)


def change_feat(f):
	def wrapper(*args):
		data = f(*args)
		if args[0].config['model'] == 'gca':
			data['state_feat'] = torch.ones(data['state_feat'].shape[0], 1)
		elif args[0].config['model'] == 'cgca_transition_condition':
			data['state_feat'] = torch.zeros(data['state_feat'].shape[0], args[0].config['z_dim'])
		return data
	return wrapper


class TransitionDataset(BaseDataset):
	def __init__(self, config: dict, mode: str):
		BaseDataset.__init__(self, config, mode)
		self.z_dim = config['z_dim']
		self.voxel_size = config['voxel_size']
		self.data_root = None
		self.data_list = []

	def cache(self, model, data, rel_file_names, step):
		batch_size = len(data['state_feat'])
		s_init = SparseTensor(
			features=torch.cat(data['state_feat']),
			coordinates=ME.utils.batched_coordinates(data['state_coord']),
			device=self.device
		)
		vis_dir = os.path.join(
			self.config['log_dir'], 'test_save',
			'step-{}'.format(step + 1), 'vis'
		)
		os.makedirs(vis_dir, exist_ok=True)
		input_imgs = tensors2tensor_imgs(
			data['state_coord'], self.config['data_dim'],
			self.config['vis']['vis_collated_imgs']['vis_3d'], batch_size
		)
		for batch_idx in range(batch_size):
			save_img_path = os.path.join(vis_dir, rel_file_names[batch_idx] + '-input.png')
			os.makedirs(os.path.dirname(save_img_path), exist_ok=True)
			save_tensor_img(input_imgs[batch_idx], save_img_path)

		if self.config['model'] == 'cgca_transition_condition':
			model.register_s0(data['input_pc'])

		for trial in range(self.config['test_trials']):
			s = s_init
			phase = Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase']
			)
			# do transition
			max_phase = self.config['max_eval_phase']
			max_ml_phase = self.config['test_mode_seeking_phase']
			for phase_cnt in range(max_phase + max_ml_phase):
				if self.config['model'] == 'gca':
					sigma = None
				else:
					sigma = model.sigma_scheduler.sample([phase])[0] \
						if phase_cnt < max_phase else None
				with torch.no_grad():
					s_next = model.transition(s, sigma)
				s = s_next
				phase += 1

			out_imgs = sparse_tensors2tensor_imgs(
				s, self.config['data_dim'],
				self.config['vis']['vis_collated_imgs']['vis_3d'], batch_size
			)
			for batch_idx in range(batch_size):
				save_tensor_img(
					out_imgs[batch_idx],
					os.path.join(
						vis_dir,
						rel_file_names[batch_idx] + '-trial{}.png'.format(trial)
					)
				)
			# cache dicts
			cache_dir = os.path.join(
				self.config['log_dir'], 'test_save',
				'step-{}'.format(step + 1), 'cache'
			)
			cache_dicts = model.sparsetensors2cache_dicts(s)
			for cache_dict, file_name in zip(cache_dicts, data['file_name']):
				save_path = os.path.join(cache_dir, file_name + '-trial={}.pt'.format(trial))
				save_dir = os.path.dirname(save_path)
				os.makedirs(save_dir, exist_ok=True)
				torch.save(cache_dict, save_path)

	def __len__(self):
		return len(self.data_list)


class TransitionCircleDataset(TransitionDataset):
	name = 'cgca_transition_circle'

	def __init__(self, config: dict, mode: str):
		TransitionDataset.__init__(self, config, mode)
		self.embedding_root = os.path.join(config['embedding_root'], mode)
		self.data_list = glob(os.path.join(self.embedding_root, '*.npz'))
		self.theta_diff = torch.tensor(np.pi / 2)
		self.surface_cnt = config['surface_cnt']

	@change_feat
	def __getitem__(self, idx):
		# load embedding data
		if self.config['overfit_one_ex'] is not None:
			idx = self.config['overfit_one_ex']
		with np.load(self.data_list[idx], 'r') as embedding:
			embedding = dict(embedding)

		embedding_coord = torch.tensor(embedding['coord'])
		embedding_feat = torch.tensor(embedding['feat'])

		# obtain initial state
		theta_start = 0 if self.mode != 'train' \
			else 2 * torch.tensor(np.pi) * torch.rand(1)

		theta = self.theta_diff * torch.rand(self.surface_cnt) + theta_start
		point_coord = torch.stack([torch.sin(theta), torch.cos(theta)], dim=1)  # tensor of N x 2

		if self.mode == 'train':
			point_coord = point_coord + embedding['translation'][:, :2]
		else:
			point_coord = point_coord + embedding['translation'][:2]
		state_coord = quantize(point_coord, self.voxel_size)
		state_feat = torch.randn(state_coord.shape[0], self.z_dim)

		return {
			'input_pc': point_coord / self.voxel_size,  # used for condition model
			'state0_coord': None,  # used for condition model
			'state0_feat': None,  # used for condition model
			'state_coord': state_coord,
			'state_feat': state_feat,
			'embedding_coord': embedding_coord,
			'embedding_feat': embedding_feat,
			'file_name': str(idx),
			'phase': Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase']
			)
		}


class TransitionShapenetDataset(TransitionDataset):
	name = 'cgca_transition_shapenet'

	def __init__(self, config: dict, mode: str):
		TransitionDataset.__init__(self, config, mode)
		self.obj_class = config['obj_class']
		self.max_sphere_centers = self.config['max_sphere_centers']
		self.sphere_radius = self.config['sphere_radius']
		self.surface_cnt = config['surface_cnt']

		if mode == 'train':
			self.data_root = os.path.join(
				config['data_root'], self.obj_class, 'train'
			)
			self.embedding_root = os.path.join(
				config['embedding_root'], self.obj_class, 'train'
			)
			data_list_file_name = 'train.txt'
		elif mode == 'val' or mode == 'test':
			self.data_root = os.path.join(
				config['data_root'], self.obj_class, 'test'
			)
			self.embedding_root = os.path.join(
				config['embedding_root'], self.obj_class, 'test'
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

	@change_feat
	def __getitem__(self, idx):
		if self.config['overfit_one_ex'] is not None:
			idx = self.config['overfit_one_ex']
		data_name = self.data_list[idx]
		data_path = os.path.join(self.data_root, data_name + '.npz')
		with np.load(data_path, 'r') as data:
			data = dict(data)

		# obtain initial state
		shape_coord = torch.tensor(data['surface'])
		if self.mode == 'train':
			num_sphere_centers = torch.randint(self.max_sphere_centers, (1,)).item() + 1
			sphere_centers = shape_coord[torch.randint(shape_coord.shape[0], (num_sphere_centers,)), :]
			if len(sphere_centers.shape) == 1:
				sphere_centers = sphere_centers.reshape(1, -1)
			survived_idxs = torch.zeros(shape_coord.shape[0]).bool()
			for center in sphere_centers:
				dists = torch.sqrt(torch.sum((shape_coord - center) ** 2, dim=1))
				survived_idxs = survived_idxs | (dists < self.sphere_radius)
			point_coord = shape_coord[survived_idxs, :]
			point_coord = downsample(point_coord, self.surface_cnt)
		else:
			point_coord = torch.tensor(data['partial'])

		# obtain embeddings
		rand_int = random.randint(0, 9)
		postfix = '.npz' if self.mode != 'train' else '_{}.npz'.format(rand_int)
		embedding_path = os.path.join(self.embedding_root, data_name + postfix)
		with np.load(embedding_path, 'r') as embedding:
			embedding = dict(embedding)

		embedding_coord = torch.tensor(embedding['coord'])
		embedding_feat = torch.tensor(embedding['feat'])

		point_coord = point_coord + embedding['translation'][:, :3]
		state_coord = quantize(point_coord, self.voxel_size)
		state_feat = torch.randn(state_coord.shape[0], self.z_dim)

		return {
			'input_pc': point_coord / self.config['voxel_size'],  # used for condition model
			'state0_coord': None,  # used for condition model
			'state0_feat': None,  # used for condition model
			'state_coord': state_coord,
			'state_feat': state_feat,
			'embedding_coord': embedding_coord,
			'embedding_feat': embedding_feat,
			'file_name': data_name,
			'phase': Phase(
				self.config['max_phase'],
				self.config['equilibrium_max_phase']
			)
		}

	def test(self, model: TransitionModel, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		# collect testset
		test_sample_nums = self.config['test_sample_nums']  # list
		testset_dict = defaultdict(list)
		partials = {}
		print('Collecting testsets...')
		for file_name in tqdm(self.data_list):
			data_path = os.path.join(self.data_root, file_name + '.npz')
			with np.load(data_path, 'r') as data:
				data = dict(data)
			partials[file_name] = torch.tensor(data['partial'])
			for sample_num in test_sample_nums:
				testset_dict[sample_num] += [data['surface'][:sample_num]]

		testset_dict = {
			k: torch.tensor(np.stack(testset_dict[k], axis=0))
			for k in testset_dict.keys()
		}
		print('Collected {} complete shapes'.format(testset_dict[list(testset_dict.keys())[0]].shape[0]))

		data_loader = DataLoader(
			self,
			batch_size=self.config['test_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
			shuffle=False
		)

		tmds = defaultdict(list)
		uhds = defaultdict(list)
		mmd_calculator = {k: MMDCalculator(testset_dict[k]) for k in testset_dict.keys()}
		for test_step, data in tqdm(enumerate(data_loader)):
			batch_size = len(data['state_feat'])
			if self.config.get('cache_only'):
				self.cache(model, data, data['file_name'], step)
			else:
				cache_dicts = model.load_cache(step, data['file_name'])
				final_pc_dict = defaultdict(list)
				for trial, cache_dicts_single_trial in enumerate(cache_dicts):
					s = model.cache_dicts2sparse_tensor(cache_dicts_single_trial)
					s_pc_dict, mesh_dict = model.get_pointcloud(s, test_sample_nums, return_mesh=True)
					for sample_num in test_sample_nums:
						final_pc_dict[sample_num].append(s_pc_dict[sample_num])

					mesh_save_dir = os.path.join(
						self.config['log_dir'], 'test_save',
						'step-{}'.format(step), 'mesh'
					)
					for k, meshes in mesh_dict.items():
						for batch_idx, mesh in enumerate(meshes):
							file_name = data['file_name'][batch_idx]
							os.makedirs(os.path.join(mesh_save_dir, k), exist_ok=True)
							mesh.export(os.path.join(mesh_save_dir, k, '{}_{}.obj'.format(file_name, trial)))

				final_pc_dict = {k: list(zip(*final_pc_dict[k])) for k in final_pc_dict.keys()}
				partial = [partials[fn].to(self.device) for fn in data['file_name']]
				for sample_num in test_sample_nums:
					for batch_idx in range(batch_size):
						pred_coords_down = torch.stack(final_pc_dict[sample_num][batch_idx], dim=0).to(self.device)
						tmds[sample_num] += [mutual_difference(pred_coords_down)]
						uhds[sample_num] += [unidirected_hausdorff_distance(partial[batch_idx], pred_coords_down)]
						mmd_calculator[sample_num].add_generated_set(pred_coords_down)
				torch.cuda.empty_cache()

		if self.config.get('cache_only') is False:
			# write to tensorboard
			for sample_num in test_sample_nums:
				mmd = mmd_calculator[sample_num].calculate_mmd()
				tmd = np.array(tmds[sample_num]).mean()
				uhd = np.array(uhds[sample_num]).mean()
				model.scalar_summaries['metrics/mmd-{}'.format(sample_num)] += [mmd]
				model.list_summaries['metrics/mmd_historgram-{}'.format(sample_num)] += mmd_calculator[sample_num].dists
				model.scalar_summaries['metrics/tmd-{}'.format(sample_num)] += [tmd]
				model.list_summaries['metrics/tmd_historgram-{}'.format(sample_num)] += tmds[sample_num]
				model.scalar_summaries['metrics/uhd-{}'.format(sample_num)] += [uhd]
				model.list_summaries['metrics/uhd_histogram-{}'.format(sample_num)] += uhds[sample_num]
				print('mmd-{}: {}\ntmd-{}: {}\nuhd-{}: {}'.format(sample_num, mmd, sample_num, tmd, sample_num, uhd))

		model.write_dict_summaries(step)
		model.train(training)


class SceneDataset(TransitionDataset):
	def test(self, model: TransitionModel, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		data_loader = DataLoader(
			self,
			batch_size=1,  # must be of batch size 1 due to varying number of sample points
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
			shuffle=False
		)

		min_chamfer_l1 = []
		mean_chamfer_l1 = []
		tmds = []
		uhds = []
		file_names = []
		chamfer_l1s = []

		for test_step, data in tqdm(enumerate(data_loader)):
			batch_size = len(data['state_feat'])
			gts = [gt.to(self.device) for gt in data['gt']]
			if self.config.get('cache_only'):
				self.cache(model, data, data['file_name'], step)
			else:
				final_pcs = []
				cache_dicts = model.load_cache(step, data['file_name'])
				test_sample_nums = [gt.shape[0] for gt in gts]
				for trial, cache_dicts_single_trial in enumerate(cache_dicts):
					s = model.cache_dicts2sparse_tensor(cache_dicts_single_trial)
					pc, mesh_dict = model.get_pointcloud(s, test_sample_nums, return_mesh=True)
					final_pcs.append(pc[list(pc.keys())[0]])

					mesh_save_dir = os.path.join(
						self.config['log_dir'], 'test_save',
						'step-{}'.format(step), 'mesh'
					)
					for k, meshes in mesh_dict.items():
						for batch_idx, mesh in enumerate(meshes):
							file_name = data['file_name'][batch_idx]
							mesh_path = os.path.join(mesh_save_dir, k, '{}_{}.obj'.format(file_name, trial))
							os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
							if not os.path.exists(mesh_path):
								mesh.export(mesh_path)

				final_pcs = list(zip(*final_pcs))
				partials = [p.to(self.device) * self.voxel_size for p in data['input_pc']]
				for batch_idx in range(batch_size):
					pred_coords_down = torch.stack(final_pcs[batch_idx], dim=0).to(self.device)
					chamfer_l1 = compute_chamfer_l1(pred_coords_down, gts[batch_idx])
					chamfer_l1 = np.array(chamfer_l1)
					min_chamfer_l1.append(chamfer_l1.min())
					mean_chamfer_l1.append(chamfer_l1.mean())
					tmds.append(mutual_difference(pred_coords_down))
					uhds.append(unidirected_hausdorff_distance(partials[batch_idx], pred_coords_down))
					chamfer_l1s.append(chamfer_l1)
				file_names.extend(data['file_name'])
				torch.cuda.empty_cache()

		if self.config.get('cache_only') is False:
			# write to tensorboard
			min_chamfer_l1 = np.array(min_chamfer_l1).mean()
			mean_chamfer_l1 = np.array(mean_chamfer_l1).mean()
			tmd = np.array(tmds).mean()
			uhd = np.array(uhds).mean()
			model.scalar_summaries['metrics/min_chamfer_l1'] += [min_chamfer_l1]
			model.scalar_summaries['metrics/mean_chamfer_l1'] += [mean_chamfer_l1]
			model.scalar_summaries['metrics/tmd'] += [tmd]
			model.list_summaries['metrics/tmd_historgram'] += tmds
			model.scalar_summaries['metrics/uhd'] += [uhd]
			model.list_summaries['metrics/uhd_histogram'] += uhds
			print(
				'min_chamfer_l1: {}\nmean_chamfer_l1: {}\ntmd: {}\nuhd: {}'.format(
					min_chamfer_l1, mean_chamfer_l1, tmd, uhd
				)
			)

			# write to file
			metrics_save_dir = os.path.join(
				self.config['log_dir'], 'test_save',
				'step-{}'.format(step),
			)
			with open(os.path.join(metrics_save_dir, 'file_names.txt'), 'w') as f:
				f.write('\n'.join(file_names))
			np.savetxt(os.path.join(metrics_save_dir, 'chamfer_l1.txt'), np.concatenate(chamfer_l1s))
			np.savetxt(os.path.join(metrics_save_dir, 'tmd.txt'), np.array(tmds))
			np.savetxt(os.path.join(metrics_save_dir, 'uhd.txt'), np.array(uhds))

		model.write_dict_summaries(step)
		model.train(training)
