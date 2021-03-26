
# standard
import numpy as np 

# custom
from solvers.solver import Solver 


class Empty(Solver):

	def __init__(self):
		pass 

	def policy(self,problem,state):
		action_dim = problem.action_dim
		return np.zeros((action_dim,1))





