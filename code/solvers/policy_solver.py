
# standard 
import numpy as np 
import torch 

# custom 
from solvers.solver import Solver 
import plotter

class PolicySolver(Solver):

	def __init__(self,policy_oracle):
		self.policy_oracle = policy_oracle
		self.solver_name = "NeuralNetwork"

	def policy(self,problem,root_state):
		action = np.zeros((problem.action_dim,1))
		action_dim_per_robot = int(problem.action_dim / problem.num_robots)
		for robot in range(problem.num_robots):
			action_idxs = np.arange(action_dim_per_robot) + action_dim_per_robot * robot
			policy_encoding = problem.policy_encoding(root_state,robot)
			policy_encoding = torch.tensor(policy_encoding,dtype=torch.float32).squeeze().unsqueeze(0)
			action[action_idxs,0] = self.policy_oracle[robot](policy_encoding).detach().numpy().squeeze()
		return action 


