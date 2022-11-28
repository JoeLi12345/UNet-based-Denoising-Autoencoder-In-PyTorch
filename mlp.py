import torch
from torch import nn
import torch.nn.functional as F

class MLP(nn.Module):
	def __init__(
		self,
		window_length=64,
		in_channels=3
	):
		"""
		Implementation of
		U-Net: Convolutional Networks for Biomedical Image Segmentation
		(Ronneberger et al., 2015)
		https://arxiv.org/abs/1505.04597
		Using the default arguments will yield the exact version used
		in the original paper
		Args:
			in_channels (int): number of input channels
			n_classes (int): number of output channels
			depth (int): depth of the network
			wf (int): number of filters in the first layer is 2**wf
			padding (bool): if True, apply padding such that the input shape
							is the same as the output.
							This may introduce artifacts
			batch_norm (bool): Use BatchNorm after layers with an
							   activation function
			up_mode (str): one of 'upconv' or 'upsample'.
						   'upconv' will use transposed convolutions for
						   learned upsampling.
						   'upsample' will use bilinear upsampling.
		"""
		super(MLP, self).__init__()
		self.input_dim = window_length*in_channels
		self.layer1 = nn.Linear(self.input_dim, 30)
		self.layer2 = nn.Linear(30, self.input_dim)
		self.relu = nn.ReLU()
		self.flatten = nn.Flatten()
		self.MLP = nn.Sequential(self.layer1, self.relu, self.layer2, self.relu)
		self.window_length = window_length
		self.in_channels = in_channels

	def forward(self, x):
		x=self.flatten(x)
		x=self.MLP(x)
		x=torch.reshape(x, (-1, self.in_channels, self.window_length))
		return x
