"""
The code for unet is mainly from Chris Choy's MinkowskiEngine (https://github.com/NVIDIA/MinkowskiEngine)
"""
import MinkowskiEngine as ME
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
import torch.nn as nn
from MinkowskiEngine import MinkowskiReLU, MinkowskiNetwork
from utils.pad import get_out_channels
import MinkowskiEngine.MinkowskiOps as me


class Res16UNetBase(MinkowskiNetwork):
	BLOCK = None
	PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
	DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
	LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
	INIT_DIM = 32
	OUT_PIXEL_DIST = 1
	# NORM_TYPE = NormType.BATCH_NORM
	# NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
	# CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

	# To use the model, must call initialize_coords before forward pass.
	# Once data is processed, call clear to reset the model before calling initialize_coords
	# def __init__(self, in_channels, out_channels, kernels, D=3):
	def __init__(self, config):
		self.config = config
		backbone_config = config['backbone']

		assert self.BLOCK is not None
		assert self.OUT_PIXEL_DIST > 0
		assert len(backbone_config['kernels']) == 10
		super().__init__(config['data_dim'])

		self.in_channels = config['backbone']['in_channels']
		model_name = config['model']
		if model_name.startswith('cgca_transition'):
			self.out_channels = config['backbone']['out_channels']
		elif model_name == 'gca':
			self.out_channels = get_out_channels(
				config['padding'], config['pad_type'],
				config['data_dim'], backbone_config['out_channels']
			)
		else:
			raise ValueError('model {} not allowed'.format(config['model']))

		self.kernels = backbone_config['kernels']
		self.D = config['data_dim']
		self.network_initialization(self.in_channels, self.out_channels, self.D)
		self.weight_initialization()

	def network_initialization(self, in_channels, out_channels, D):
		# Setup net_metadata
		dilations = self.DILATIONS
		bn_momentum = 0.02

		def space_n_time_m(n, m):
			return n if D <= 3 else [n, n, n, m]

		if D == 4:
			self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

		# Output of the first conv concated to conv6
		self.inplanes = self.INIT_DIM
		self.conv0p1s1 = ME.MinkowskiConvolution(
			in_channels, self.inplanes,
			kernel_size=self.kernels[0], dimension=D
		)
		self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)
		self.conv1p1s2 = ME.MinkowskiConvolution(
			self.inplanes, self.inplanes,
			kernel_size=self.kernels[1], stride=2, dimension=D
		)

		self.bn1 = ME.MinkowskiBatchNorm(self.inplanes)
		self.block1 = self._make_layer(
			self.BLOCK, self.PLANES[0], self.LAYERS[0]
		)

		self.conv2p2s2 = ME.MinkowskiConvolution(
			self.inplanes, self.inplanes,
			kernel_size=self.kernels[2], stride=2, dimension=D
		)
		self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

		self.block2 = self._make_layer(
			self.BLOCK, self.PLANES[1], self.LAYERS[1]
		)

		self.conv3p4s2 = ME.MinkowskiConvolution(
			self.inplanes, self.inplanes,
			kernel_size=self.kernels[3], stride=2, dimension=D
		)

		self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
		self.block3 = self._make_layer(
			self.BLOCK, self.PLANES[2], self.LAYERS[2]
		)

		self.conv4p8s2 = ME.MinkowskiConvolution(
			self.inplanes, self.inplanes,
			kernel_size=self.kernels[4], stride=2, dimension=D
		)
		self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
		self.block4 = self._make_layer(
			self.BLOCK, self.PLANES[3], self.LAYERS[3]
		)

		self.convtr4p16s2 = ME.MinkowskiConvolutionTranspose(
			self.inplanes, self.PLANES[4],
			kernel_size=self.kernels[5], stride=2, dimension=D
		)
		self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

		self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
		self.block5 = self._make_layer(
			self.BLOCK, self.PLANES[4], self.LAYERS[4]
		)
		self.convtr5p8s2 = ME.MinkowskiConvolutionTranspose(
			self.inplanes, self.PLANES[5],
			kernel_size=self.kernels[6], stride=2, dimension=D
		)
		self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

		self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
		self.block6 = self._make_layer(
			self.BLOCK, self.PLANES[5], self.LAYERS[5]
		)
		self.convtr6p4s2 = ME.MinkowskiConvolutionTranspose(
			self.inplanes, self.PLANES[6],
			kernel_size=self.kernels[7], stride=2, dimension=D
		)
		self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

		self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
		self.block7 = self._make_layer(self.BLOCK, self.PLANES[6],
									   self.LAYERS[6])
		self.convtr7p2s2 = ME.MinkowskiConvolutionTranspose(
			self.inplanes, self.PLANES[7],
			kernel_size=self.kernels[8], stride=2, dimension=D
		)
		self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

		self.inplanes = self.PLANES[7] + self.INIT_DIM
		self.block8 = self._make_layer(
			self.BLOCK, self.PLANES[7], self.LAYERS[7]
		)

		self.final = ME.MinkowskiConvolution(
			self.PLANES[7] * self.BLOCK.expansion,
			out_channels,
			kernel_size=self.kernels[9],
			bias=True,
			dimension=D
		)
		self.relu = ME.MinkowskiReLU(inplace=True)


	def weight_initialization(self):
		for m in self.modules():
			if isinstance(m, ME.MinkowskiBatchNorm):
				nn.init.constant_(m.bn.weight, 1)
				nn.init.constant_(m.bn.bias, 0)

	def _make_layer(
			self, block, planes, blocks,
			stride=1, dilation=1, bn_momentum=0.1
	):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				ME.MinkowskiConvolution(
					self.inplanes, planes * block.expansion,
					kernel_size=1, stride=stride, dimension=self.D
				),
				ME.MinkowskiBatchNorm(planes * block.expansion))
		layers = []
		layers.append(
			block(
				self.inplanes, planes, stride=stride,
				dilation=dilation, downsample=downsample, dimension=self.D
			)
		)
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(
				block(
					self.inplanes, planes,
					stride=1, dilation=dilation, dimension=self.D
				)
			)

		return nn.Sequential(*layers)

	def forward(self, x, out_coord=None):
		out = self.conv0p1s1(x)
		out = self.bn0(out)
		out_p1 = self.relu(out)
		out = self.conv1p1s2(out_p1)
		out = self.bn1(out)
		out = self.relu(out)
		out_b1p2 = self.block1(out)

		out = self.conv2p2s2(out_b1p2)
		out = self.bn2(out)
		out = self.relu(out)
		out_b2p4 = self.block2(out)

		out = self.conv3p4s2(out_b2p4)
		out = self.bn3(out)
		out = self.relu(out)
		out_b3p8 = self.block3(out)

		# pixel_dist=16
		out = self.conv4p8s2(out_b3p8)

		out = self.bn4(out)
		out = self.relu(out)
		out = self.block4(out)

		# pixel_dist=8
		out = self.convtr4p16s2(out)
		out = self.bntr4(out)
		out = self.relu(out)

		out = me.cat(out, out_b3p8)
		out = self.block5(out)

		# pixel_dist=4
		out = self.convtr5p8s2(out)
		out = self.bntr5(out)
		out = self.relu(out)

		out = me.cat(out, out_b2p4)
		out = self.block6(out)

		# pixel_dist=2
		out = self.convtr6p4s2(out)
		out = self.bntr6(out)
		out = self.relu(out)

		out = me.cat(out, out_b1p2)
		out = self.block7(out)

		# pixel_dist=1
		out = self.convtr7p2s2(out)
		out = self.bntr7(out)
		out = self.relu(out)

		out = me.cat(out, out_p1)
		out = self.block8(out)

		return self.final(out, coordinates=out_coord)


class Res16UNet14(Res16UNetBase):
	BLOCK = BasicBlock
	LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


class Res16UNet18(Res16UNetBase):
	BLOCK = BasicBlock
	LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


class Res16UNet34(Res16UNetBase):
	BLOCK = BasicBlock
	LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet50(Res16UNetBase):
	BLOCK = Bottleneck
	LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


class Res16UNet101(Res16UNetBase):
	BLOCK = Bottleneck
	LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


class Res16UNet14A(Res16UNet14):
	PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet14A2(Res16UNet14A):
	LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B(Res16UNet14):
	PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet14B2(Res16UNet14B):
	LAYERS = (1, 1, 1, 1, 2, 2, 2, 2)


class Res16UNet14B3(Res16UNet14B):
	LAYERS = (2, 2, 2, 2, 1, 1, 1, 1)


class Res16UNet14C(Res16UNet14):
	name = 'Mink16UNet14C'
	PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


class Res16UNet14D(Res16UNet14):
	name = 'Mink16UNet14D'
	PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet18A(Res16UNet18):
	name = 'Mink16UNet18A'
	PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


class Res16UNet18B(Res16UNet18):
	name = 'Mink16UNet18B'
	PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


class Res16UNet18D(Res16UNet18):
	PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


class Res16UNet34A(Res16UNet34):
	PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


class Res16UNet34B(Res16UNet34):
	PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


class Res16UNet34C(Res16UNet34):
	name = 'Mink16UNet34C'
	PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

