
# standard 
import numpy as np 

# custom 
from problems.problem import POSG
from problems.types.space import Cube

# 
t0 = 0 
tf = 10 
dt = 0.1
times = np.arange(t0,tf,dt)

# 
num_robots = 2
state_dim_per_robot = 7
action_dim_per_robot = 3
state_dim = state_dim_per_robot * num_robots
action_dim = action_dim_per_robot * num_robots

# reward 
Q = np.zeros((state_dim_per_robot,state_dim_per_robot))
Q[0,0] = 1
Q[1,1] = 1
Q[2,2] = 1
d = 0.2
Ru = 0.1*np.eye(action_dim_per_robot)

# state and action lim 
pos_lim = 2 # m 
rad_lim = 4*np.pi # rad 
vel_lim = 2  # m / s
omega_lim = 2*np.pi / 10 # rad / s 
acc_lim = 1.0 # m / s^2


class Example2(POSG):

	def __init__(self): 

		S = self.make_S()
		A = self.make_A()
		O = None 
		Z = None 
		R = self.reward() 
		T = None 
		b0 = None 
		gamma = 1.0 

		super(Example2,self).__init__(S,A,O,Z,R,T,b0,gamma)
		self.dt = dt
		self.times = times 


	def is_terminal(self,state):
		return not self.is_valid(state)


	def is_valid(self,state):
		return self.S.contains(state) 


	def reward(self,s,a):
		s_1 = s[0:state_dim_per_robot]
		s_2 = s[state_dim_per_robot:]
		reward = -1 * (
			np.abs((s_1-s_2)^T @ Q @ (s_1 - s_2) - d) + \
			a_1^T @ Ru @ a_1)
		return reward 


	def normalized_reward(self,s,a):
		# needs to be in [0,1]
		reward = self.reward(s,a)
		r_max = 100
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)


	def make_S(self):
		state_lims = np.zeros((state_dim,2))
		for i in range(num_robots):
			shift = i * state_dim_per_robot 
			state_lims[shift+0:3,0] = -pos_lim 
			state_lims[shift+0:3,1] = pos_lim 
			state_lims[shift+3:6,0] = -rad_lim 
			state_lims[shift+3:6,1] = rad_lim 
			state_lims[shift+6,0] = -vel_lim
			state_lims[shift+6,1] = vel_lim 		
		return Cube(state_lims)


	def make_A(self):
		action_lims = np.zeros((action_dim,2))
		for i in range(num_robots):
			shift = i * action_dim_per_robot 
			action_lims[shift+0:3,0] = -pos_lim 
			state_lims[shift+0:3,1] = pos_lim 
			state_lims[shift+3:6,0] = -rad_lim 
			state_lims[shift+3:6,1] = rad_lim 
			state_lims[shift+6,0] = -vel_lim
			state_lims[shift+6,1] = vel_lim 
		return Cube(action_lims)


