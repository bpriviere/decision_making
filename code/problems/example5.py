


# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d double integrator , single robot, action includes timestep 
class Example5(Problem):

	def __init__(self): 
		super(Example5,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.gamma = 1.0
		self.mass = 1
		self.num_robots = 1 
		self.state_dim_per_robot = 4 
		self.action_dim_per_robot = 3
		self.r_max = 100
		self.state_control_weight = 1.0
		self.name = "example5"
		self.position_idx = np.arange(2) 

		self.state_dim = self.num_robots * self.state_dim_per_robot
		self.action_dim = self.num_robots * self.action_dim_per_robot
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array([
			[-5,5],
			[-5,5],
			[-1,1],
			[-1,1]
		])

		self.dm_dt = 0.05
		self.action_lims = np.array([
			[-1,1],
			[-1,1],
			[1*self.dm_dt,10*self.dm_dt],
		])

		self.init_lims = np.array([
			[-5,5],
			[-5,5],
			[-1,1],
			[-1,1]
		])
		self.F = np.eye(self.state_dim) + self.dt * np.array((
			(0,0,1,0),
			(0,0,0,1),
			(0,0,0,0),
			(0,0,0,0)
			))
		self.B = self.dt / self.mass * np.array((
			(0,0,0),
			(0,0,0),
			(1,0,0),
			(0,1,0),
			))

		self.Q = np.eye(self.state_dim)
		self.Ru = self.state_control_weight * np.eye(self.action_dim)
		self.Ru[2,2] = 0 

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a)))
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		r_max = self.r_max
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)

	def step(self,s,a):
		t = 0 
		curr_state = s 

		F = np.eye(self.state_dim) + self.dm_dt * np.array((
			(0,0,1,0),
			(0,0,0,1),
			(0,0,0,0),
			(0,0,0,0)
			))
		B = self.dm_dt / self.mass * np.array((
			(0,0,0),
			(0,0,0),
			(1,0,0),
			(0,1,0),
			))

		while t < a[2]:
			curr_state = np.dot(F,curr_state) + np.dot(B,a)
			t = t + self.dm_dt 
		return curr_state

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

