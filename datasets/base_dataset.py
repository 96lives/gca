import torch
import random
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from models.base_model import Model
from torch.utils.tensorboard import SummaryWriter
from typing import List


class BaseDataset(Dataset, ABC):
	name = 'base'

	def __init__(self, config: dict, mode: str = 'train'):
		self.config = config
		self.mode = mode
		self.device = config['device']
		self.data_dim = config['data_dim']
		self.summary_name = self.name

	'''
	Note that dataset's __getitem__() returns (x_coord, x_feat, y_coord, y_feat, name)
	But the collated batch returns type of (SparseTensorWrapper, SparseTensorWrapper)
	'''
	def __getitem__(self, idx) \
			-> (torch.tensor, torch.tensor, torch.tensor, torch.tensor, List[str]):
		# sparse tensor and tensor should have equal size
		raise NotImplemented

	def __iter__(self):
		while True:
			idx = random.randint(0, len(self) - 1)
			yield self[idx]

	def collate_fn(self, batch: List) -> dict:
		# convert list of dict to dict of list
		batch = {k: [d[k] for d in batch] for k in batch[0]}
		return batch

	def evaluate(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()
		data_loader = DataLoader(
			self,
			batch_size=self.config['eval_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=False,
		)

		print('')
		eval_losses = []
		for eval_step, data in enumerate(data_loader):
			mode = self.mode
			if len(self.config['eval_datasets']) != 1:
				mode += '_' + self.summary_name
			eval_loss = model.evaluate(data, step, mode)
			eval_losses.append(eval_loss)

			print('\r[Evaluating, Step {:7}, Loss {:5}]'.format(
				eval_step, '%.3f' % eval_loss), end=''
			)

		print('')
		model.write_dict_summaries(step)
		model.train(training)

	def test(self, model: Model, writer: SummaryWriter, step):
		raise NotImplementedError()

	def visualize(self, model: Model, options: List, step):
		training = model.training
		model.eval()

		# fix vis_indices
		vis_indices = self.config['vis']['indices']
		if isinstance(vis_indices, int):
			# sample data points from n data points with equal interval
			n = len(self)
			vis_indices = torch.linspace(0, n - 1, vis_indices).int().tolist()

		# override to the index when in overfitting debug mode
		if isinstance(self.config['overfit_one_ex'], int):
			vis_indices = torch.tensor([self.config['overfit_one_ex']])

		for option in options:
			# calls the visualizing function
			if hasattr(model, option):
				getattr(model, option)(self, vis_indices, step)
			else:
				raise ValueError(
					'model {} has no method {}'.format(
						model.__class__.__name__, option
					)
				)
		model.train(training)

	def visualize_test(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		# fix vis_indices
		vis_indices = self.config['vis']['indices']
		if isinstance(vis_indices, int):
			# sample data points from n data points with equal interval
			vis_indices = torch.linspace(0, len(self) - 1, vis_indices).int().tolist()

		# override to the index when in overfitting debug mode
		if isinstance(self.config['overfit_one_ex'], int):
			vis_indices = torch.tensor([self.config['overfit_one_ex']])

		model.visualize_test(self, vis_indices, step)
		model.train(training)

