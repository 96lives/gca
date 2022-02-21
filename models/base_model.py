import torch
import numpy as np
import os
import psutil
import gc
from torch.utils.tensorboard import SummaryWriter
from torch import nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
from time import time
from typing import List


# ==========
# Model ABCs
# ==========


class Model(nn.Module, ABC):
	def __init__(self, config, writer: SummaryWriter):
		nn.Module.__init__(self)
		self.config = config
		self.device = config['device']
		self.writer = writer
		self.backbone = None  # initialize on concrete model
		self.step_time = time()
		self.loss = None
		self.scalar_summaries = defaultdict(list)
		self.list_summaries = defaultdict(list)
		self.data_dim = config['data_dim']


	def init_pretrained(self, pretrained_path, strict):
		state_dict = torch.load(pretrained_path)
		if state_dict.get('state_dict') is not None:
			state_dict = state_dict['state_dict']
		# remove final
		if not strict:
			for k in list(state_dict.keys()):
				if 'final' in k:
					state_dict.pop(k)
		self.backbone.load_state_dict(state_dict, strict=strict)

	def _clip_grad_value(self, clip_value):
		for group in self.optimizer.param_groups:
			nn.utils.clip_grad_value_(group['params'], clip_value)

	def _clip_grad_norm(self, max_norm, norm_type=2):
		for group in self.optimizer.param_groups:
			nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

	def clip_grad(self):
		clip_grad_config = self.config['clip_grad']
		if clip_grad_config['type'] == 'value':
			self._clip_grad_value(**clip_grad_config['options'])
		elif clip_grad_config['type'] == 'norm':
			self._clip_grad_norm(**clip_grad_config['options'])
		else:
			raise ValueError('Invalid clip_grad type: {}'
							 .format(clip_grad_config.type))

	def get_lr(self):
		for param_group in self.optimizer.param_groups:
			return param_group['lr']

	def write_dict_summaries(self, step):

		# write scalar summaries
		for (k, v) in self.scalar_summaries.items():
			v = np.array(v).mean().item()
			self.writer.add_scalar(k, v, step)

			if k == 'phase/incomplete_cnt':
				num_complete = self.list_summaries.get('completion_phase/train')
				num_complete = len(num_complete) if num_complete is not None else 0
				complete_rate = num_complete / (v + num_complete)
				self.writer.add_scalar('phase/complete_rate', complete_rate, step)
				self.writer.add_scalar('phase/complete_cnt', num_complete, step)

		# write list summaries
		for (k, v) in self.list_summaries.items():
			self.writer.add_histogram(k, np.array(v), step)
		# reset summaries
		self.scalar_summaries.clear()
		self.list_summaries.clear()

	def write_summary(self, scheduler, step):
		# write summaries
		# write all the averaged summaries
		self.write_dict_summaries(step)

		# write current learning rate
		self.writer.add_scalar('lr', self.get_lr(), step)

		# write resources
		# write time elapsed since summary_step
		resource_prefix = 'resources/'
		self.writer.add_scalar(
			resource_prefix + 'time_per_step',
			(time() - self.step_time) / self.config['summary_step'], step
		)
		self.step_time = time()

		self.writer.add_scalar(
			resource_prefix + 'gpu_memory',
			torch.cuda.max_memory_reserved(self.device) / float(2 ** 30), step
		)

		process = psutil.Process(os.getpid())
		self.writer.add_scalar(
			resource_prefix + 'memory',
			process.memory_info().rss / float(2 ** 30), step
		)
		torch.cuda.empty_cache()
		gc.collect()

		if self.config['model'] in ['cgca_transition', 'gca']:
			self.writer.add_scalar(
				resource_prefix + 'buffer/num_elements',
				len(scheduler.data_buffer.buffer), step
			)
			num_voxels = 0
			for data in scheduler.data_buffer.buffer:
				num_voxels += data['state_coord'].shape[0]
			self.writer.add_scalar(
				resource_prefix + 'buffer/num_voxels',
				num_voxels, step
			)
			self.writer.add_scalar(
				resource_prefix + 'buffer/removal_cnt',
				scheduler.data_buffer.buffer_removal_cnt, step
			)

	@abstractmethod
	def forward(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def learn(self, *args, **kwargs):
		raise NotImplementedError

	@abstractmethod
	def evaluate(self, data: dict, step: int, mode: str):
		raise NotImplementedError



