
# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator , single robot 
class Example1(Problem):

	def __init__(self): 
		super(Example1,self).__init__()

		self.t0 = 0
		self.tf = 2
		self.dt = 0.1
		self.r_max = 100
		self.num_robots = 1
		self.gamma = 1
		self.state_dim = 2
		self.action_dim = 2
		self.state_control_weight = 1.0
		self.name = "example1"
		self.position_idx = np.arange(2)

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim
		self.state_dim_per_robot = self.state_dim
		self.action_dim_per_robot = self.action_dim

		self.state_lims = np.array([
			[-5,5],
			[-5,5]
		])
		self.action_lims = np.array([
			[-1,1],
			[-1,1]
		])
		self.init_lims = np.array([
			[-5,5],
			[-5,5]
		])

		self.F = np.eye(self.state_dim) 
		self.B = self.dt * np.eye(self.state_dim)
		self.Q = np.eye(self.state_dim)
		self.Ru = self.state_control_weight * np.eye(self.action_dim)

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a))).squeeze()
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		r_max = self.r_max
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)

	def step(self,s,a):
		s_tp1 = np.dot(self.F,s) + np.dot(self.B,a)
		return s_tp1 

	def render(self,states,fig=None,ax=None):
		if fig == None or ax == None: 
			fig,ax = plotter.make_fig()	
		states = np.array(states)
		state_lims = self.state_lims
		ax.plot(states[:,0],states[:,1])
		ax.plot(states[0,0],states[0,1],'o')
		ax.plot(states[-1,0],states[-1,1],'s')
		ax.set_xlim([state_lims[0,0],state_lims[0,1]])
		ax.set_ylim([state_lims[1,0],state_lims[1,1]])
		return fig,ax

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 