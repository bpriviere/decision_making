
# standard 
import numpy as np 
import matplotlib.pyplot as plt 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# dummy game for debugging: multiple single integrator regulator problems 
class Example10(Problem):

	def __init__(self): 
		super(Example10,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.r_max = 100
		self.r_min = -100
		self.num_robots = 2
		self.gamma = 0.99

		self.state_idxs = [[] for _ in range(self.num_robots)]
		self.action_idxs = [[] for _ in range(self.num_robots)]
		for robot in range(self.num_robots):
			state_shift = 2 * robot 
			action_shift = 2 * robot 
			self.state_idxs[robot] = state_shift + np.arange(2)
			self.action_idxs[robot] = action_shift + np.arange(2)

		self.state_dim = 2 * self.num_robots 
		self.action_dim = 2 * self.num_robots 
		self.state_control_weight = 1.0
		self.name = "example10"
		self.position_idx = np.arange(2)

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.zeros((2 * self.num_robots, 2))
		self.state_lims[:,0] = -5 
		self.state_lims[:,1] = 5 

		eps = 0.00001
		self.action_lims = np.array((
			[-eps,eps],
			[1-eps,1],
			[1-eps,1],
			[-eps,eps]
			))

		# self.action_lims = np.zeros((2 * self.num_robots, 2))
		# self.action_lims[:,0] = -1
		# self.action_lims[:,1] = 1

		self.init_lims = self.state_lims

		self.Fc = np.zeros((2,2))
		self.Bc = np.eye(2)
		self.Q = np.eye(2)
		self.Ru = self.state_control_weight * np.eye(2)

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		for robot in range(self.num_robots):
			si = s[self.state_idxs[robot],:]
			ai = a[self.action_idxs[robot],:]
			reward[robot,0] = -1 * (np.dot(si.T,np.dot(self.Q,si)) + np.dot(ai.T,np.dot(self.Ru,ai))).squeeze()
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		reward = np.clip(reward,self.r_min,self.r_max)
		return (reward - self.r_min) / (self.r_max - self.r_min)

	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		Fd = np.eye(2) + self.Fc * dt 
		Bd = self.Bc * dt 
		for robot in range(self.num_robots):
			s_tp1[self.state_idxs[robot],:] = np.dot(Fd,s[self.state_idxs[robot],:]) + np.dot(Bd,a[self.action_idxs[robot],:])
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		if fig == None or ax == None: 
			fig,ax = plotter.make_fig()	

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)

			for robot in range(self.num_robots):
				ax.plot(states[:,self.state_idxs[robot][0]],states[:,self.state_idxs[robot][1]],color=colors[robot])
				ax.plot(states[0,self.state_idxs[robot][0]],states[0,self.state_idxs[robot][1]],'o',color=colors[robot])
				ax.plot(states[-1,self.state_idxs[robot][0]],states[-1,self.state_idxs[robot][1]],'s',color=colors[robot])
			
			state_lims = self.state_lims
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
		ax.quiver(state[:,self.state_idxs[robot][0]],state[:,self.state_idxs[robot][1]],action[:,0],action[:,1])
		ax.set_title("{} Policy for Robot {}".format(title,robot))
		ax.set_xlim(self.state_lims[0,:])
		ax.set_ylim(self.state_lims[1,:])


	def plot_value_dataset(self,dataset,title):

		state = dataset[0] 
		value = dataset[1] 

		# contour
		for robot in range(self.num_robots):
			fig,ax = plt.subplots()
			pcm = ax.tricontourf(state[:,self.state_idxs[robot][0]],state[:,self.state_idxs[robot][1]],value[:,robot])
			ax.set_title("{} Value for Robot {}".format(title,robot))
			ax.set_xlim(self.state_lims[0,:])
			ax.set_ylim(self.state_lims[1,:])

