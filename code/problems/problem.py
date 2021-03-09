
# standard 
import numpy as np 

# custom
from problems.types.space import Cube

class Problem: 

	def __init__(self):
		pass 

	def reward(self,state,action):
		exit("reward needs to be overwritten")

	def step(self,state,action):
		exit("step needs to be overwritten")

	def render(self,state):
		exit("render needs to be overwritten")

	def is_terminal(self,state):
		exit("is_terminal needs to be overwritten")


class MDP(Problem):

	def __init__(self,S,A,R,T,gamma):
		# input: 
		# 	- S: state space, elements in [state_dim x 1]
		# 	- A: action space, elements in [action_dim x 1]
		# 	- R: reward function, r = R(s,a)
		# 	- T: transition function, P(s_tp1 | s_t,a_t) = T(s_tp1,s_t,a_t)
		# 	- gamma: discount factor 
		self.S = S 
		self.A = A
		self.reward = R
		self.T = T
		self.gamma = gamma
		self.num_robots = 1
		super(MDP, self).__init__()


class LQR(MDP):
	
	def __init__(self,F,B,Q,Ru):
		self.F = F 
		self.B = B 
		self.Q = Q 
		self.Ru = Ru 
		self.state_dim,self.action_dim = np.shape(B)
		S = self.make_S() 
		A = self.make_A() # action space 
		R = self.reward 
		T = None
		gamma = 1
		super(LQR, self).__init__(S,A,R,T,gamma)

	def step(self,s,a):
		# input: 
		# 	- numpy float [state_dim x 1]
		# 	- numpy float [action_dim x 1]
		# output: 
		# 	- numpy float [state_dim x 1]
		s_tp1 = np.dot(self.F,s) + np.dot(self.B,a)
		return s_tp1 

	def reward(self,s,a):
		# output:
		# 	- numpy float scalar 
		return -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a))).squeeze()


class POMDP(Problem):

	def __init__(self,S,A,O,Z,R,T,b0,gamma):
		# input: 
		# 	- S: state space, elements in [state_dim x 1]
		# 	- A: action space, elements in [action_dim x 1]
		# 	- O: observation function, P(z|s,a) = O(s,a,z)
		# 	- Z: observation space, elements in [measurement_dim x 1]
		# 	- R: reward function, r = R(s,a)
		# 	- T: transition function, P(s_tp1 | s_t,a_t) = T(s_tp1,s_t,a_t)
		# 	- b0: initial beliefs, P(b0|s0)
		# 	- gamma: discount factor 
		self.num_robots = 1
		super(POMDP, self).__init__()		


class POSG(Problem):

	def __init__(self,param,S,A,O,Z,R,T,b0,gamma,I):
		# input: 
		# 	- S: state space for each agent, dict of space objects 
		# 	- A: action space for each agent, dict of space objects
		# 	- O: observation function for each agent, dict of O_i functions, P(z|s,a) = O_i(s,a,z)
		# 	- Z: observation space, [num agents x measurement dim]
		# 	- R: reward function for each agent, dict of functions, r_i = R_i(s,a)
		# 	- T: transition function for each agent, dict of functions, P(s_{t+1} | s_t,a_t) = T
		# 	- I: index set of robots 
		self.num_robots = len(I)
		super(POSG, self).__init__()