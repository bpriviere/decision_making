
# standard package
import torch

# my package
from learning.feedforward import FeedForward

class PolicyNetwork(torch.nn.Module):

	def __init__(self,problem,device="cpu",path=None):
		super(PolicyNetwork, self).__init__()

		h = 12

		self.encoding_dim = problem.policy_encoding_dim
		self.output_dim = 2*int(problem.action_dim/problem.num_robots) 
		self.state_dim = problem.state_dim 
		self.action_dim = problem.action_dim 
		self.device = torch.device(device)

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

		dist = self.psi(x)
		split = int(dist.shape[1] / 2)
		mu = dist[:,0:split]
		logvar = dist[:,split:]

		if training:
			return mu, logvar
		else:
			batch_size = x.shape[0]
			sd = torch.sqrt(torch.exp(logvar))
			eps = torch.randn(size=(batch_size,int(self.output_dim/2)),device=self.device)
			policy = mu + sd * eps
			return policy 
			