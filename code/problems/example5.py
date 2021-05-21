


# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# game of attrition and attack - see page 97 of Isaacs book 
class Example5(Problem):

	def __init__(self): 
		super(Example5,self).__init__()
		# states:
		# 	- s[0]: player 1's number of forces 
		# 	- s[1]: player 2's "" 
		# actions: 
		# 	- a[0]: player 1's (1-fraction of forces) devoted to offensive
		# 	- a[1]: player 2's "" 
		# rewards: 
		# 	- r[0]: player 1's margin of superiority over one day
		# 	- r[1]: player 2's ""  

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.gamma = 0.99
		self.num_robots = 2 
		self.state_dim = 2
		self.action_dim = 2
		self.r_max = 1
		self.r_min = -1 * self.r_max
		self.name = "example5"
		self.position_idx = np.arange(1) 
		self.state_control_weight = 0 # not used 

		# problem specific parameters 
		self.m1 = 1 # manufacturing capacity 
		self.m2 = 1
		self.c1 = 1 # measure of effectiveness  
		self.c2 = 1 

		self.state_idxs = [np.arange(1), 1+np.arange(1)] 
		self.action_idxs = [np.arange(1), 1+np.arange(1)] 
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			( 0,20),
			( 0,20),
			))

		self.action_lims = np.array((
			( 0,1), # phi
			( 0,1), # psi
			))

		self.init_lims = np.array((
			( 0,10),
			( 0,10),
			))

	def initialize(self):
		valid = False
		while not valid:
			state = sample_vector(self.init_lims)
			valid = self.is_valid(state)
		# start with same initial number of forces 
		# state[1] = state[0]
		return state

	def reward(self,s,a):
		r = (1-a[1,0])*s[1,0] - (1-a[0,0])*s[0,0] 
		reward = np.array([[-r],[r]])
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		reward = np.clip(reward,self.r_min,self.r_max)
		reward = (reward - self.r_min) / (self.r_max - self.r_min)
		reward = np.array([[reward[0,0]],[1-reward[0,0]]])
		return reward

	def step(self,s,a,dt):
		sdot = np.zeros(s.shape)
		sdot[0,0] = self.m1 - self.c1 * a[1,0] * s[1,0]
		sdot[1,0] = self.m2 - self.c2 * a[0,0] * s[0,0]

		s_tp1 = np.zeros(s.shape)
		s_tp1 = s + sdot * dt 
		return s_tp1

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		lims = self.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_xlabel("$x_1$")
		ax.set_ylabel("$x_2$")

		if states is not None:
			ax.plot(states[:,0], states[:,1])
			ax.plot(states[0,0], states[0,1],marker='o')
			ax.plot(states[-1,0], states[-1,1],marker='s')
			
		return fig,ax 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 

	def plot_value_dataset(self,dataset,title):
		pass 

	def plot_policy_dataset(self,dataset,title,robot):
		pass 

	def pretty_plot(self,sim_result):
		pass 