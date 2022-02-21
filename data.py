import random
from collections import Iterator, defaultdict
from torch.utils.data import DataLoader
from utils.util import timeit
from datasets import DATASET

# =====================
# Base Classes and ABCs
# =====================


class DataScheduler(Iterator):
	def __init__(self, config):
		self.config = config
		self.dataset = DATASET[self.config['dataset']](config, mode='train')
		self.eval_datasets = [
			DATASET[x[0]](config, mode=x[1])
			for x in self.config['eval_datasets']
		]

		if self.config.get('test_datasets') is not None:
			self.test_datasets = [
				DATASET[x[0]](config, mode=x[1])
				for x in self.config['test_datasets']
			]
		self.total_epoch = self.config['epoch']
		self.step_cnt = 0
		self.epoch_cnt = 0
		self._remainder = len(self.dataset)
		self.data_loader = DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			shuffle=True
		)
		self.iter = iter(self.data_loader)
		self._check_vis = {}

		self.use_buffer = config['model'] in [
			'gca',
			'cgca_transition',
			'cgca_transition_condition',
			'cgca_transition_connection',
		]
		if self.use_buffer:
			self.data_buffer = DataBuffer(config)

	@timeit
	def __next__(self):
		'''
		:return:
			data: dict of corresponding data
		'''
		if self.data_loader is None:
			raise StopIteration

		if self.use_buffer:
			while self.data_buffer.is_full() is False:
				try:
					data = next(self.iter)
				except StopIteration:
					self.iter = iter(self.data_loader)
					data = next(self.iter)
				self.data_buffer.push(data)
				self.update_epoch_cnt()
			data = self.data_buffer.sample(self.config['batch_size'])
		else:
			# used for training patch_autoencoder
			try:
				data = next(self.iter)
			except StopIteration:
				self.iter = iter(self.data_loader)
				data = next(self.iter)
			self.update_epoch_cnt()
		self.step_cnt += 1
		return data, self.epoch_cnt

	def __len__(self):
		return len(self.sampler)

	def check_eval_step(self, step):
		if (step + 1) < self.config['min_eval_step']:
			return False
		return ((step + 1) % self.config['eval_step'] == 0) \
			   or self.config['debug_eval']

	def check_test_step(self, step):
		if (step + 1) < self.config['min_test_step']:
			return False

		return (step + 1) % self.config['test_step'] == 0 \
			if self.config.get('test_step') is not None else False

	def check_vis_step(self, step):
		if (step + 1) < self.config['min_vis_step']:
			return False

		vis = False
		vis_config = self.config['vis']
		for (k, v) in vis_config.items():
			# check if valid visualization config
			if not isinstance(v, dict):
				continue
			if ((step + 1) % v['step'] == 0) or (self.config['debug_vis']):
				self._check_vis[k] = True
				vis = True
			else:
				self._check_vis[k] = False
		return vis

	def check_summary_step(self, step):
		return (step + 1) % self.config['summary_step'] == 0

	def check_empty_cache_step(self, step):
		if self.config.get('empty_cache_step') is None:
			return False
		return (step + 1) % self.config['empty_cache_step'] == 0

	def evaluate(self, model, writer, step):
		for eval_dataset in self.eval_datasets:
			eval_dataset.evaluate(model, writer, step)

	def test(self, model, writer, step):
		print('Testing...')
		if self.test_datasets is not None:
			for test_dataset in self.test_datasets:
				test_dataset.test(model, writer, step)

	def visualize_test(self, model, writer, step):
		self.test_dataset.visualize_test(model, writer, step)

	def visualize(self, model, writer, step):

		# find options to visualize in this step
		options = []
		for (k, v) in self._check_vis.items():
			if not v:
				continue
			else:
				options.append(k)

		if isinstance(self.config['overfit_one_ex'], int):
			self.dataset.visualize(model, options, step)
		else:
			self.dataset.visualize(model, options, step)  # train dataset
			for eval_dataset in self.eval_datasets:  # eval dataset
				eval_dataset.visualize(model, options, step)
		# reset _check_vis
		self._check_vis = {}

	def update_epoch_cnt(self):
		self._remainder -= self.config['batch_size']
		if self._remainder < self.config['batch_size']:
			self._remainder += len(self.dataset)
			self.epoch_cnt += 1


class DataBuffer:
	def __init__(self, config):
		self.config = config
		self.buffer_size = config['buffer_size']
		self.buffer = []
		self.device = config['device']
		self.max_batch_points = config['mean_vox_points'] * config['batch_size']
		self.buffer_removal_cnt = 0

	def push(self, data):
		phase = data['phase']
		for batch_idx in range(len(phase)):
			if phase[batch_idx].finished:
				continue
			coord_cnt = data['state_coord'][batch_idx].shape[0]
			max_coord_cnt = self.config.get('voxel_overflow') \
				if self.config.get('voxel_overflow') is not None else 1000000000000
			if (coord_cnt == 0) or (coord_cnt > max_coord_cnt):
				self.buffer_removal_cnt += 1
				continue
			self.buffer.append({
				k: v[batch_idx]
				for k, v in data.items()
			})

	def sample(self, batch_size):
		data = defaultdict(list)
		cum_batch_points = 0
		for batch_idx in range(batch_size):
			idx = random.randint(0, len(self.buffer) - 1)
			# assure that the # coords in batch is not too big
			if ((cum_batch_points + self.buffer[idx]['state_coord'].shape[0]) > self.max_batch_points) \
					and (batch_idx > 0):
				break
			pop_data = self.buffer.pop(idx)
			for k, v in pop_data.items():
				data[k].append(pop_data[k])
			cum_batch_points += pop_data['state_coord'].shape[0]
		# convert back to dict so that no accidents occur
		return dict(data)

	def is_full(self):
		return len(self.buffer) >= self.buffer_size


