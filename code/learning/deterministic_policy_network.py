# standard package
import torch
from torch.nn import MSELoss

# my package
from learning.feedforward import FeedForward

class DeterministicPolicyNetwork(torch.nn.Module):

	def __init__(self,problem,device="cpu",path=None,output_dim=None):
		super(DeterministicPolicyNetwork, self).__init__()

		self.encoding_dim = problem.policy_encoding_dim
		if output_dim is None:
			output_dim = int(problem.action_dim / problem.num_robots)
		self.output_dim = output_dim
		self.path = path

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


	def eval(self,problem,root_state,robot):
		policy_encoding = problem.policy_encoding(root_state,robot)
		policy_encoding = torch.tensor(policy_encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		policy = self.__call__(policy_encoding).detach().numpy().reshape(self.output_dim,1) # [action_dim_per_robot x 1]
		return policy 