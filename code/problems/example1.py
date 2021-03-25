
# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator , single robot 
class Example1(Problem):

	def __init__(self,\
		t0 = 0,
		tf = 10,
		dt = 0.1,
		pos_lim = 5,
		vel_lim = 1,
		): 
		super(Example1,self).__init__()

		times = np.arange(t0,tf,dt)
		
		Fc = np.array(((0,0),(0,0)))
		Bc = np.eye(2)
		position_idx = np.arange(2)

		state_dim,action_dim = Bc.shape
		state_lims = np.zeros((state_dim,2))
		for i_s in range(state_dim):
			state_lims[i_s,0] = -pos_lim
			state_lims[i_s,1] =  pos_lim
		action_lims = np.zeros((action_dim,2))
		for i_s in range(action_dim):
			action_lims[i_s,0] = -vel_lim
			action_lims[i_s,1] =  vel_lim

		self.F = np.eye(state_dim) +  Fc * dt 
		self.B = Bc * dt
		self.Q = np.eye(state_dim)
		self.Ru = np.eye(action_dim)

		# 
		self.num_robots = 1
		self.gamma = 1
		self.state_dim = state_dim
		self.state_lims = state_lims 
		self.action_dim = action_dim
		self.action_lims = action_lims 
		self.position_idx = position_idx 
		self.dt = dt
		self.times = times  
		self.policy_encoding_dim = state_dim
		self.value_encoding_dim = state_dim

	def sample_action(self):
		return sample_vector(self.action_lims)

	def sample_state(self):
		return sample_vector(self.state_lims)

	def reward(self,s,a):
		return -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a))).squeeze()		

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		r_max = 100
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)

	def step(self,s,a):
		s_tp1 = np.dot(self.F,s) + np.dot(self.B,a)
		return s_tp1 

	def render(self,states):
		states = np.array(states)
		state_lims = self.state_lims
		fig,ax = plotter.make_fig() 
		ax.plot(states[:,0],states[:,1])
		ax.plot(states[0,0],states[0,1],'o')
		ax.plot(states[-1,0],states[-1,1],'s')
		ax.set_xlim([state_lims[0,0],state_lims[0,1]])
		ax.set_ylim([state_lims[1,0],state_lims[1,1]])

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def initialize(self):
		return self.sample_state()

	def steer(self,s1,s2):
		num_samples = 10 
		for i in range(num_samples):
			action = self.sample_action()
			dist = self.dist(self.step(s1,action),s2)
			if i == 0 or dist < best_dist:
				best_action = action
				best_dist = dist 
		return best_action 

	def dist(self,s1,s2):
		return np.linalg.norm(s1-s2)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 