import torch.nn as nn
import torch
import copy

class FeedForward(nn.Module):

	def __init__(self,layers,activation,device):
		super(FeedForward, self).__init__()

		self.layers = nn.ModuleList()
		for layer in layers:
			if layer[0] == "Linear":
				self.layers.append(nn.Linear(layer[1], layer[2]))
			else:
				raise Exception("Unknown layer type: {}".format(layer[0]))

		if activation == "relu":
			self.activation = torch.relu
		else:
			raise Exception("Unknown activation: {}".format(activation))

		self.device = torch.device(device)
		self.in_dim = self.layers[0].in_features
		self.out_dim = self.layers[-1].out_features

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		x = self.layers[-1](x)
		return x

	def export_to_onnx(self, filename):
		dummy_input = torch.randn(self.in_dim)
		torch.onnx.export(self, dummy_input, "{}.onnx".format(filename), export_params=True, keep_initializers_as_inputs=True)
