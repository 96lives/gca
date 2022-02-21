import MinkowskiEngine as ME
import torch.nn as nn
from MinkowskiEngine import MinkowskiReLU, MinkowskiNetwork, MinkowskiMaxPooling


class SparseIFNet(MinkowskiNetwork):
	def __init__(self, config):
		self.config = config

		super().__init__(config['data_dim'])

		self.D = config['data_dim']
		self.z_dim = config['z_dim']

		self.conv0s1 = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=3, dimension=self.D
		)
		self.bn0 = ME.MinkowskiBatchNorm(self.z_dim)
		self.conv0s1_out = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=3, dimension=self.D
		)

		self.conv1s2 = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=2, stride=2, dimension=self.D
		)
		self.bn1 = ME.MinkowskiBatchNorm(self.z_dim)
		self.conv1s1_out = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=3, dimension=self.D
		)

		self.conv2s2 = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=3, stride=2, dimension=self.D
		)
		self.bn2 = ME.MinkowskiBatchNorm(self.z_dim)
		self.conv2s1_out = ME.MinkowskiConvolution(
			self.z_dim, self.z_dim,
			kernel_size=3, dimension=self.D
		)
		self.relu = ME.MinkowskiReLU(inplace=True)
		self.pooling = MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=self.D)

		self.weight_initialization()

	def weight_initialization(self):
		for m in self.modules():
			if isinstance(m, ME.MinkowskiBatchNorm):
				nn.init.constant_(m.bn.weight, 1)
				nn.init.constant_(m.bn.bias, 0)

	def forward(self, x):
		out = ME.SparseTensor(x.F, x.C)
		out = self.conv0s1(out)
		out = self.bn0(out)
		out = self.relu(out)
		out_p1 = self.conv0s1_out(out)

		out = self.conv1s2(out)
		out = self.bn1(out)
		out = self.relu(out)
		out_p2 = self.conv1s1_out(out)

		out = self.conv2s2(out)
		out = self.bn1(out)
		out = self.relu(out)
		out_p3 = self.conv2s1_out(out)

		return out_p1, out_p2, out_p3
