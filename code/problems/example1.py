
# standard 
import numpy as np 
import matplotlib.pyplot as plt 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator , single robot 
class Example1(Problem):

	def __init__(self): 
		super(Example1,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.r_max = 100
		self.r_min = -100
		self.num_robots = 1
		self.gamma = 0.99
		self.state_dim = 2
		self.action_dim = 2
		self.state_control_weight = 1.0
		self.name = "example1"
		self.position_idx = np.arange(2)

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim
		self.state_idxs = [np.arange(self.state_dim)]
		self.action_idxs = [np.arange(self.action_dim)]

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

		self.Fc = np.zeros((self.state_dim,self.state_dim))
		self.Bc = np.eye(self.state_dim)
		self.Q = np.eye(self.state_dim)
		self.Ru = self.state_control_weight * np.eye(self.action_dim)

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a))).squeeze()
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		reward = np.clip(reward,self.r_min,self.r_max)
		return (reward - self.r_min) / (self.r_max - self.r_min)

	def step(self,s,a,dt):
		Fd = np.eye(self.state_dim) + self.Fc * dt 
		Bd = self.Bc * dt 
		s_tp1 = np.dot(Fd,s) + np.dot(Bd,a)
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		if fig == None or ax == None: 
			fig,ax = plotter.make_fig()	
		if states is not None:
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

	def plot_policy_dataset(self,dataset,title,robot):

		state = dataset[0] 
		action = dataset[1] 

		# quiver 
		fig,ax = plt.subplots()
		ax.quiver(state[:,0],state[:,1],action[:,0],action[:,1])
		ax.set_title("{} Policy for Robot {}".format(title,robot))
		ax.set_xlim(self.state_lims[0,:])
		ax.set_ylim(self.state_lims[1,:])


	def plot_value_dataset(self,dataset,title):

		state = dataset[0] 
		value = dataset[1] 

		# quiver 
		fig,ax = plt.subplots()
		pcm = ax.tricontourf(state[:,0],state[:,1],value[:,0])
		ax.set_title("{} Value".format(title))
		ax.set_xlim(self.state_lims[0,:])
		ax.set_ylim(self.state_lims[1,:])


