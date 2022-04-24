import os
import glob
import random
from PIL import Image
import numpy as np
import trimesh
from baselines.data.core import Field
# from baselines.utils import binvox_rw
# from baselines.common import coord2index, normalize_coord


class IndexField(Field):
    ''' Basic index field.'''
    def load(self, model_path, idx, category, obj_min_rate):
        ''' Loads the index field.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        return idx

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        return True

# 3D Fields
# class PatchPointsField(Field):
#     ''' Patch Point Field.
#
#     It provides the field to load point data. This is used for the points
#     randomly sampled in the bounding volume of the 3D shape and then split to patches.
#
#     Args:
#         file_name (str): file name
#         transform (list): list of transformations which will be applied to the points tensor
#         multi_files (callable): number of files
#
#     '''
#     def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
#         self.file_name = file_name
#         self.transform = transform
#         self.unpackbits = unpackbits
#         self.multi_files = multi_files
#
#     def load(self, model_path, idx, vol, obj_min_rate):
#         ''' Loads the data point.
#
#         Args:
#             model_path (str): path to model
#             idx (int): ID of data point
#             vol (dict): precomputed volume info
#         '''
#         if self.multi_files is None:
#             file_path = os.path.join(model_path, self.file_name)
#         else:
#             num = np.random.randint(self.multi_files)
#             file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
#
#         points_dict = np.load(file_path)
#         points = points_dict['points']
#         # Break symmetry if given in float16:
#         if points.dtype == np.float16:
#             points = points.astype(np.float32)
#             points += 1e-4 * np.random.randn(*points.shape)
#
#         occupancies = points_dict['occupancies']
#         if self.unpackbits:
#             occupancies = np.unpackbits(occupancies)[:points.shape[0]]
#         occupancies = occupancies.astype(np.float32)
#
#         # acquire the crop
#         ind_list = []
#         for i in range(3):
#             ind_list.append((points[:, i] >= vol['query_vol'][0][i])
#                      & (points[:, i] <= vol['query_vol'][1][i]))
#         ind = ind_list[0] & ind_list[1] & ind_list[2]
#         data = {None: points[ind],
#                     'occ': occupancies[ind],
#             }
#
#         if self.transform is not None:
#             data = self.transform(data)
#
#         # calculate normalized coordinate w.r.t. defined query volume
#         p_n = {}
#         for key in vol['plane_type']:
#             # projected coordinates normalized to the range of [0, 1]
#             p_n[key] = normalize_coord(data[None].copy(), vol['input_vol'], plane=key)
#         data['normalized'] = p_n
#
#         return data

class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the points
    randomly sampled in the bounding volume of the 3D shape.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor
        multi_files (callable): number of files

    '''
    def __init__(self, file_name, transform=None, unpackbits=False, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.unpackbits = unpackbits
        self.multi_files = multi_files

    def load(self, model_path, idx, category, obj_min_rate):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        points_dict = np.load(file_path)
        points = points_dict['points']
        # Break symmetry if given in float16:
        if points.dtype == np.float16:
            points = points.astype(np.float32)
            points += 1e-4 * np.random.randn(*points.shape)

        occupancies = points_dict['occupancies']
        if self.unpackbits:
            occupancies = np.unpackbits(occupancies)[:points.shape[0]]
        occupancies = occupancies.astype(np.float32)

        data = {
            None: points,
            'occ': occupancies,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

# class VoxelsField(Field):
#     ''' Voxel field class.
#
#     It provides the class used for voxel-based data.
#
#     Args:
#         file_name (str): file name
#         transform (list): list of transformations applied to data points
#     '''
#     def __init__(self, file_name, transform=None):
#         self.file_name = file_name
#         self.transform = transform
#
#     def load(self, model_path, idx, category, obj_min_rate):
#         ''' Loads the data point.
#
#         Args:
#             model_path (str): path to model
#             idx (int): ID of data point
#             category (int): index of category
#         '''
#         file_path = os.path.join(model_path, self.file_name)
#
#         with open(file_path, 'rb') as f:
#             voxels = binvox_rw.read_as_3d_array(f)
#         voxels = voxels.data.astype(np.float32)
#
#         if self.transform is not None:
#             voxels = self.transform(voxels)
#
#         return voxels
#
#     def check_complete(self, files):
#         ''' Check if field is complete.
#
#         Args:
#             files: files
#         '''
#         complete = (self.file_name in files)
#         return complete


# class PatchPointCloudField(Field):
#     ''' Patch point cloud field.
#
#     It provides the field used for patched point cloud data. These are the points
#     randomly sampled on the mesh and then partitioned.
#
#     Args:
#         file_name (str): file name
#         transform (list): list of transformations applied to data points
#         multi_files (callable): number of files
#     '''
#     def __init__(self, file_name, transform=None, transform_add_noise=None, multi_files=None):
#         self.file_name = file_name
#         self.transform = transform
#         self.multi_files = multi_files
#
#     def load(self, model_path, idx, vol, obj_min_rate):
#         ''' Loads the data point.
#
#         Args:
#             model_path (str): path to model
#             idx (int): ID of data point
#             vol (dict): precomputed volume info
#         '''
#         if self.multi_files is None:
#             file_path = os.path.join(model_path, self.file_name)
#         else:
#             num = np.random.randint(self.multi_files)
#             file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))
#
#         pointcloud_dict = np.load(file_path)
#
#         points = pointcloud_dict['points'].astype(np.float32)
#         normals = pointcloud_dict['normals'].astype(np.float32)
#
#         # add noise globally
#         if self.transform is not None:
#             data = {None: points,
#                     'normals': normals}
#             data = self.transform(data)
#             points = data[None]
#
#         # acquire the crop index
#         ind_list = []
#         for i in range(3):
#             ind_list.append((points[:, i] >= vol['input_vol'][0][i])
#                     & (points[:, i] <= vol['input_vol'][1][i]))
#         mask = ind_list[0] & ind_list[1] & ind_list[2]# points inside the input volume
#         mask = ~mask # True means outside the boundary!!
#         data['mask'] = mask
#         points[mask] = 0.0
#
#         # calculate index of each point w.r.t. defined resolution
#         index = {}
#
#         for key in vol['plane_type']:
#             index[key] = coord2index(points.copy(), vol['input_vol'], reso=vol['reso'], plane=key)
#             if key == 'grid':
#                 index[key][:, mask] = vol['reso']**3
#             else:
#                 index[key][:, mask] = vol['reso']**2
#         data['ind'] = index
#
#         return data
#
#     def check_complete(self, files):
#         ''' Check if field is complete.
#
#         Args:
#             files: files
#         '''
#         complete = (self.file_name in files)
#         return complete

class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the mesh.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
    '''
    def __init__(self, file_name, transform=None, multi_files=None):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files

    def load(self, model_path, idx, category, obj_min_rate=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)
        bboxes = np.load(os.path.join(model_path, 'item_dict.npz'), allow_pickle=True)['bboxes']

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)

        if obj_min_rate is not None:
            # erase objects
            surface_removed = []
            normals_removed = []
            is_obj = np.zeros(points.shape[0]).astype(np.bool8)
            ground = points[:, 1].min()
            for i in range(bboxes.shape[0]):
                bbox = bboxes[i]
                left_bound = (points[:, 0] >= bbox[0][0]) & (points[:, 2] >= bbox[0][1])
                right_bound = (points[:, 0] <= bbox[1][0]) & (points[:, 2] <= bbox[1][1])
                selected = left_bound & right_bound & (points[:, 1] > ground)
                obj_surface = points[selected]
                survived_idxs = np.ones(obj_surface.shape[0]).astype(np.bool8)
                original_idxs = survived_idxs

                for _ in range(20):
                    center_idx = np.random.randint(obj_surface.shape[0], size=1).item()
                    center = obj_surface[center_idx, :].reshape(-1, 3)
                    dists = np.sqrt(np.sum((obj_surface - center) ** 2, axis=1))
                    temp_survived_idxs = survived_idxs & (dists > 0.1)
                    num_current_points = temp_survived_idxs.sum()
                    if num_current_points < (obj_min_rate * original_idxs.shape[0]):
                        continue
                    survived_idxs = temp_survived_idxs
                is_obj = is_obj | selected

                surface_removed.append(obj_surface[survived_idxs, :])
                normals_removed.append(normals[selected][survived_idxs, :])

            # erase wall
            wall_surface = points[~is_obj]
            survived_idxs = np.ones(wall_surface.shape[0]).astype(np.bool8)
            original_idxs = survived_idxs
            for _ in range(20):
                center_idx = np.random.randint(wall_surface.shape[0], size=1).item()
                center = wall_surface[center_idx, :].reshape(-1, 3)
                dists = np.sqrt(np.sum((wall_surface - center) ** 2, axis=1))
                temp_survived_idxs = survived_idxs & (dists > 0.1)
                num_current_points = temp_survived_idxs.sum()
                if num_current_points < (0.8 * original_idxs.shape[0]):
                    continue
                survived_idxs = temp_survived_idxs
            surface_removed.append(wall_surface[survived_idxs, :])
            normals_removed.append(normals[~is_obj][survived_idxs, :])

            points = np.concatenate(surface_removed, axis=0)
            normals = np.concatenate(normals_removed, axis=0)

        data = {
            None: points,
            'normals': normals,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete


class PartialPointCloudField(Field):
    ''' Partial Point cloud field.

    It provides the field used for partial point cloud data. These are the points
    randomly sampled on the mesh and a bounding box with random size is applied.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        multi_files (callable): number of files
        part_ratio (float): max ratio for the remaining part
    '''
    def __init__(self, file_name, transform=None, multi_files=None, part_ratio=0.7):
        self.file_name = file_name
        self.transform = transform
        self.multi_files = multi_files
        self.part_ratio = part_ratio

    def load(self, model_path, idx, category, obj_min_rate):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
        '''
        if self.multi_files is None:
            file_path = os.path.join(model_path, self.file_name)
        else:
            num = np.random.randint(self.multi_files)
            file_path = os.path.join(model_path, self.file_name, '%s_%02d.npz' % (self.file_name, num))

        pointcloud_dict = np.load(file_path)

        points = pointcloud_dict['points'].astype(np.float32)
        normals = pointcloud_dict['normals'].astype(np.float32)


        side = np.random.randint(3)
        xb = [points[:, side].min(), points[:, side].max()]
        length = np.random.uniform(self.part_ratio*(xb[1] - xb[0]), (xb[1] - xb[0]))
        ind = (points[:, side]-xb[0])<= length
        data = {
            None: points[ind],
            'normals': normals[ind],
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.file_name in files)
        return complete
