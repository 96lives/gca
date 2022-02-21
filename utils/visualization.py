from mpl_toolkits.mplot3d import Axes3D
from torchvision.transforms import ToTensor, Resize, ToPILImage
from typing import List, Union, Tuple, Dict
from MinkowskiEngine import SparseTensor

import io
import torch
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt


def save_tensor_img(img, path):
	img = ToPILImage()(img)
	img.save(path)


def plt_to_tensor(plt, h, w, clear=True):
	buf = io.BytesIO()
	plt.savefig(buf, dpi=np.mean([h, w]), format='png')
	if clear:
		plt.clf()
	buf.seek(0)
	img = PIL.Image.open(buf)
	img = Resize((h, w))(img)
	return ToTensor()(img)


def vis_3d_coords(
		x: Union[
			SparseTensor, torch.Tensor,
			List[torch.Tensor],
			Tuple[torch.Tensor],
		], img_config, batch_size,
		scene_capture=False, losses=None,
		num_views=1, max_sample=100000
) \
		-> (List[torch.Tensor], List):
	'''
	Args
		x:
			SparseTensor or
			Tensor of size [B, N, 3] or
			Tuple of 2 tensors containing query coordinates and occupancies
		max_sample:
			maximum number of samples
		num_view:
			number of views to visualize in plot
		axis_ranges:
			list of [x_lim, y_lim, z_lim]
			used to make the scale of axis consistent
	:return:
		imgs: list of img (in tensor)
		new_axis_range: [x_lim, y_lim, z_lim]
			x_lim: range of img's x axes (tuple of min max)
			y_lim: range of img's y axes (tuple of min max)
			z_lim: range of img's z axes (tuple of min max)
	'''
	imgs = []

	for batch_idx in range(batch_size):
		# Create Image Figure
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		# Set axis range for plotting
		if img_config.get('axis_ranges') is not None:
			ax.set_xlim(img_config['axis_ranges'][0])
			ax.set_ylim(img_config['axis_ranges'][1])
			ax.set_zlim(img_config['axis_ranges'][2])

		if (type(x) == torch.Tensor) \
				or (type(x) == list) or (type(x) == tuple):
			coord = x[batch_idx].detach().cpu()
		elif type(x) == SparseTensor:
			try:
				coord = x.coords_at(batch_idx).detach().cpu()
			except RuntimeError:
				# empty coordinates
				coord = torch.zeros(1, 3)
		else:
			raise ValueError('type {} not supported in visualization'.format(type(x)))

		if max_sample > 0:
			sample = np.random.RandomState(0)\
				.permutation(coord.shape[0])[:max_sample]
			coord = coord[sample]

		plt.axis('off')
		alpha = 0.05 if scene_capture else img_config['alpha']

		ax.scatter(
			xs=coord[:, 0], ys=coord[:, 2], zs=coord[:, 1],
			alpha=alpha, marker='o', linewidths=0
		)
		if losses != None:
			ax.set_title("Loss: {0:.3f}".format(losses[batch_idx]), fontsize=24)
		single_img = []
		init_angle = -45
		for angle in range(init_angle, init_angle + 360, int(360 / num_views)):
			ax.view_init(elev=None, azim=angle)
			single_img.append(
				plt_to_tensor(
					plt, img_config['height'], img_config['width'], clear=False
				)
			)
		imgs.append(torch.cat(single_img, dim=1))
		plt.close('all')

	return imgs


def vis_2d_coords(
		coords: List[torch.Tensor], img_config: Dict,
		feats: Union[List[torch.Tensor], None] = None
)\
		-> List[torch.Tensor]:
	"""
	:param coords: List of tensors of dim N x 2
	:param img_config: Dictionary containing
		- h: height of output tensor
		- w: width of output tensor
		- alpha: tranparency
		- axis_ranges:
			list of [x_lim, y_lim]
			used to make the scale of axis consistent
	:param feats: None or list of tensors 
	:return: 
		list of tensors (imgs) of h x w
	"""
	h, w = img_config['height'], img_config['width']
	axis_ranges, alpha = img_config.get('axis_ranges'), img_config['alpha']

	imgs = []
	for batch_idx in range(len(coords)):
		coord = coords[batch_idx].detach().cpu().numpy()
		plt.figure()
		plt.xlabel('X')
		plt.ylabel('Y')

		if axis_ranges is not None:
			plt.xlim(axis_ranges[0])
			plt.ylim(axis_ranges[1])

		try:
			if feats is None:
				plt.scatter(
					x=coord[:, 0], y=coord[:, 1],
					marker='o', alpha=alpha, linewidths=0
				)
			else:
				feat = feats[batch_idx].detach().cpu().numpy()
				plt.scatter(
					x=coord[:, 0], y=coord[:, 1], c=feat,
					cmap='Paired', vmin=0, vmax=1,
					marker='o', alpha=alpha, linewidths=0
				)
				plt.colorbar(
					label='sdf', aspect=10,
					boundaries=np.linspace(0, 1, 13), spacing='uniform'
				)
		except:
			breakpoint()
		imgs.append(plt_to_tensor(plt, h, w, clear=True))
	plt.close('all')
	return imgs

def tensors2tensor_imgs(
		x: Union[
			torch.Tensor,
			List[torch.Tensor],
			Tuple[torch.Tensor],
		],
		data_dim, img_config,
		batch_size, losses=None
) -> List[torch.Tensor]:

	if data_dim == 2:
		tensor_imgs = vis_2d_coords(
			x, img_config=img_config,
		)
	elif data_dim == 3:
		num_views = img_config.get('num_views') \
			if img_config.get('num_views') is not None else 1
		tensor_imgs = vis_3d_coords(
			x, img_config, batch_size,
			scene_capture=img_config['scene_capture'],
			num_views=num_views, losses=losses
		)
	else:
		raise NotImplementedError()

	return tensor_imgs


def sparse_tensors2tensor_imgs(
		x: SparseTensor,
		data_dim, img_config, batch_size,
		losses=None,
) -> List[torch.Tensor]:

	# add empty zero tensor if sparse tensor is empty
	tensor_list = []
	for batch_idx in range(batch_size):
		idx = x.C[:, 0] == batch_idx
		if idx.sum() != 0:
			tensor_list.append(x.C[idx, 1:])
		else:
			tensor_list.append(torch.zeros(1, data_dim))

	if data_dim == 2:
		tensor_imgs = vis_2d_coords(
			tensor_list, img_config=img_config,
		)
	elif data_dim == 3:
		num_views = img_config.get('num_views') \
			if img_config.get('num_views') is not None else 1
		tensor_imgs = vis_3d_coords(
			tensor_list, img_config, batch_size,
			scene_capture=img_config['scene_capture'],
			num_views=num_views, losses=losses
		)
	else:
		raise NotImplementedError

	return tensor_imgs


def tensors2dist_func_tensor_imgs(
		x: List[torch.Tensor], img_config,
		feats: Union[List[torch.Tensor], None] = None,
) -> List[torch.Tensor]:
	x_new, feats_new = [], []
	try:
		project_axis = img_config['project_axis']
		if project_axis == 'x':
			axis = 0
		elif project_axis == 'y':
			axis = 1
		elif project_axis == 'z':
			axis = 2
		else:
			raise ValueError('axis {} not allowed'.format(project_axis))
		low = img_config['project_center'] - img_config['project_thres']
		high = img_config['project_center'] + img_config['project_thres']
		if feats is None:
			feats_new = None
			for batch_idx, x_single in enumerate(x):
				idx = (x_single[:, axis] < high) & (x_single[:, axis] > low)
				x_new.append(x_single[idx].detach().cpu())
		else:
			for batch_idx, (x_single, feat) in enumerate(zip(x, feats)):
				idx = (x_single[:, axis] < high) & (x_single[:, axis] > low)
				x_new.append(x_single[idx].detach().cpu())
				feats_new.append(feat[idx].detach().cpu())
	except:
		breakpoint()

	tensor_dist_imgs = vis_2d_coords(
		x_new, img_config=img_config,
		feats=feats_new
	)

	return tensor_dist_imgs

