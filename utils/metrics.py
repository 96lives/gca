import torch
import numpy as np
import os
import warnings
from utils.chamfer_distance import ChamferDistance
from typing import List
from sklearn.neighbors import NearestNeighbors
from scipy.stats import entropy

chamfer_dist = ChamferDistance()


def compute_chamfer_dist(c1, c2, no_sqrt=True):
	with torch.no_grad():
		dist1, dist2 = chamfer_dist(c1, c2)

	if no_sqrt:
		return dist1.mean().item() + dist2.mean().item()
	else:
		return (dist1.sqrt().mean().item() + dist2.sqrt().mean().item()) / 2


class MMDCalculator:
	INF = 9999999.

	def __init__(self, testset):
		self.testset = testset.cpu()
		self.test_size = testset.shape[0]
		self.min_dists = torch.ones(self.test_size).float() * MMDCalculator.INF

	def add_generated_set(self, preds: torch.tensor):
		'''
		Args:
			preds: torch tensor of {test_trials} x {test_pred_downsample} x 3
			testset: torch tensor of {self.test_size} x {test_pred_downsample} x 3
		'''
		for pred in preds:
			# reshape tensor to {test_size} x {test_pred_downsample} x 3
			pred = pred.unsqueeze(0).expand(self.test_size, -1, -1)
			with torch.no_grad():
				dist1, dist2 = chamfer_dist(pred, self.testset.to(pred.device))
			dists = dist1.mean(dim=1).cpu() + dist2.mean(dim=1).cpu()  # {test_size}
			self.min_dists = torch.where(dists < self.min_dists, dists, self.min_dists)
		self.testset = self.testset.cpu()

	def calculate_mmd(self) -> float:
		return self.min_dists.mean().item()

	@property
	def dists(self) -> List[float]:
		return self.min_dists.tolist()

	def reset(self):
		self.min_dists = torch.ones(self.test_size).float() * MMDCalculator.INF


def compute_chamfer_l1(pred_set, gt):
	'''
	:param point_cloud1: torch.tensor of modality x M x 3 tensor
	:param point_cloud2: torch.tensor M x 3
	:return: list of chamfer l1 distances
	'''
	return [
		compute_chamfer_dist(
			pred.unsqueeze(0).float(),
			gt.unsqueeze(0).float(),
			no_sqrt=True
		) for pred in pred_set
	]


def mutual_difference(inferred_set):
	'''
	inferred_set : modality x M x 3 tensor (modality = 10, M = 2048)
	'''
	inferred_set = inferred_set.view(inferred_set.shape[0], 1, inferred_set.shape[1], -1)

	md = 0
	for j in range(inferred_set.shape[0]):
		for l in range(j + 1, inferred_set.shape[0], 1):
			md += compute_chamfer_dist(inferred_set[j], inferred_set[l], no_sqrt=True)

	return 2 * md / (inferred_set.shape[0] - 1)


def directed_hausdorff(point_cloud1, point_cloud2, reduce_mean=False):
	"""
	point_cloud1: (B, N, 3) torch tensor
	point_cloud2: (B, M, 3) torch tensor
	return: directed hausdorff distance, pc1 -> pc2
	"""
	n_pts1 = point_cloud1.shape[1]
	n_pts2 = point_cloud2.shape[1]
	pc1 = torch.transpose(point_cloud1, 1, 2)  # (B, 3, N)
	pc2 = torch.transpose(point_cloud2, 1, 2)  # (B, 3, M)
	pc1 = pc1.unsqueeze(3)
	pc1 = pc1.repeat((1, 1, 1, n_pts2))
	pc2 = pc2.unsqueeze(2)
	pc2 = pc2.repeat((1, 1, n_pts1, 1))

	l2_dist = torch.sqrt(torch.sum((pc1 - pc2) ** 2, dim=1))
	shortest_dist, _ = torch.min(l2_dist, dim=2)
	hausdorff_dist, _ = torch.max(shortest_dist, dim=1)

	if reduce_mean:
		hausdorff_dist = torch.mean(hausdorff_dist)

	return hausdorff_dist.item()


def directed_hausdorff_chamfer(point_cloud1, point_cloud2, reduce_mean=False):
	"""
	point_cloud1: (B, N, 3) torch tensor
	point_cloud2: (B, M, 3) torch tensor
	return: directed hausdorff distance, pc1 -> pc2
	"""
	dist1, dist2 = chamfer_dist(point_cloud1, point_cloud2)
	return dist1.max().sqrt().item()


def unidirected_hausdorff_distance(partial_input, pred_set, use_chamfer=True):
	'''
	Args:
		partial_input: torch.tensor of shape {test_input_downsample} x 3
		pred_set: torch.tensor of {trials} x {test_pred_downsample} x 3

	Returns:
		float output of the calculated score
	'''
	partial_input = partial_input.view(1, partial_input.shape[0], -1)
	pred_set = pred_set.view(pred_set.shape[0], 1, pred_set.shape[1], -1)
	uhd = 0
	for i in range(partial_input.shape[0]):
		if use_chamfer:
			uhd += directed_hausdorff_chamfer(partial_input, pred_set[i])
		else:
			uhd += directed_hausdorff(partial_input, pred_set[i])

	return uhd / pred_set.shape[0]



