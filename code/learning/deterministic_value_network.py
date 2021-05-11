# standard package
import torch
from torch.nn import MSELoss

# my package
from learning.feedforward import FeedForward

class DeterministicValueNetwork(torch.nn.Module):

	def __init__(self,problem,device="cpu",path=None):
		super(DeterministicValueNetwork, self).__init__()

		self.encoding_dim = problem.value_encoding_dim
		self.output_dim = problem.num_robots
		self.path = path
		self.name = "deterministic"

		self.state_dim = problem.state_dim 
		self.action_dim = problem.action_dim 
		self.device = torch.device(device)

		h = 12
		psi_network_architecture = [
			["Linear", self.encoding_dim, h], 
			["Linear", h, h],
			["Linear", h, self.output_dim] 
		]

		network_activation = "relu"

		self.psi = FeedForward(
			psi_network_architecture,
			network_activation,
			device)	

		self.to(self.device)

		if path is not None: 
			self.load_state_dict(torch.load(path))


	def to(self, device):
		self.device = device
		self.psi.to(device)
		return super().to(device)


	def __call__(self,x,training=False):
		return self.psi(x)


	def loss_fnc(self,x,target):
		criterion = MSELoss()
		y = self.__call__(x)
		loss = criterion(y,target)
		return loss 


	def eval(self,problem,root_state):
		value_encoding = problem.value_encoding(root_state)
		value_encoding = torch.tensor(value_encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		value = self.__call__(value_encoding).detach().numpy().reshape(problem.num_robots,1) # [num_robots x 1]
		return value 