
# standard 
import numpy as np 
import torch 

# custom 
from solvers.solver import Solver 
from learning.policy_network import PolicyNetwork
from learning.value_network import ValueNetwork
import plotter

class PolicySolver(Solver):

	def __init__(self,problem,paths):
		self.models = [PolicyNetwork(problem,path=path) for path in paths]
		
	def policy(self,problem,root_state):
		action = np.zeros((problem.action_dim,1))
		action_dim_per_robot = int(problem.action_dim / problem.num_robots)
		for robot in range(problem.num_robots):
			action_idxs = np.arange(action_dim_per_robot) + action_dim_per_robot * robot
			policy_encoding = problem.policy_encoding(root_state,robot)
			policy_encoding = torch.tensor(policy_encoding,dtype=torch.float32).squeeze().unsqueeze(0)
			action[action_idxs,0] = self.models[robot](policy_encoding).detach().numpy().squeeze()
		return action 

class ValueSolver(Solver):

	def __init__(self,problem,path):
		self.model = ValueNetwork(problem,path=path)
		
	def __call__(self,problem,root_state):
		value_encoding = problem.value_encoding(root_state)
		value_encoding = torch.tensor(value_encoding,dtype=torch.float32).squeeze().unsqueeze(0)
		value = self.model(value_encoding).detach().numpy().squeeze()
		return value 	
