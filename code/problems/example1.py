
# standard 
import numpy as np 

# custom 
from problems.problem import LQR
from problems.types.space import Cube

state_dim = 2 
action_dim = 2 
t0 = 0 
tf = 10 
dt = 0.1
times = np.arange(t0,tf,dt)

# dynamics
Fc = np.zeros((state_dim,state_dim))
Bc = np.eye(state_dim)
F = np.eye(state_dim) +  Fc * dt 
B = Bc * dt
Q = np.eye(state_dim)
Ru = np.eye(action_dim)

# state and action lim 
pos_lim = 20
action_lim = 5


class Example1(LQR):

	def __init__(self): 
		super(Example1,self).__init__(F,B,Q,Ru)
		self.dt = dt
		self.times = times 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return self.S.contains(state)

	def normalized_reward(self,s,a):
		# needs to be in [0,1]
		reward = self.reward(s,a)
		r_max = 100
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)

	def make_S(self):
		state_lims = np.zeros((self.state_dim,2))
		for i_s in range(self.state_dim):
			state_lims[i_s,0] = -pos_lim
			state_lims[i_s,1] =  pos_lim 			
		return Cube(state_lims)

	def make_A(self):
		action_lims = np.zeros((self.action_dim,2))
		for i_a in range(self.action_dim):
			action_lims[i_a,0] = -action_lim
			action_lims[i_a,1] =  action_lim
		return Cube(action_lims)		


