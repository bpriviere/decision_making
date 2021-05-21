


# standard 
import numpy as np 
import matplotlib.pyplot as plt 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d double integrator , single robot 
class Example2(Problem):

	def __init__(self): 
		super(Example2,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.gamma = 1.0
		self.mass = 1
		self.num_robots = 1 
		self.state_dim = 4 
		self.action_dim = 2 
		self.r_max = 100
		self.r_min = -1*self.r_max
		self.state_control_weight = 1.0
		self.name = "example2"
		self.position_idx = np.arange(2) 

		self.state_idxs = [np.arange(self.state_dim)]
		self.action_idxs = [np.arange(self.action_dim)]
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array([
			[-5,5],
			[-5,5],
			[-1,1],
			[-1,1]
		])
		self.action_lims = np.array([
			[-1,1],
			[-1,1]
		])
		self.init_lims = np.array([
			[-5,5],
			[-5,5],
			# [-1,1],
			# [-1,1]
			[0,0],
			[0,0]
		])
		self.Fc = np.array((
			(0,0,1,0),
			(0,0,0,1),
			(0,0,0,0),
			(0,0,0,0)
			))
		self.Bc = 1.0 / self.mass * np.array((
			(0,0),
			(0,0),
			(1,0),
			(0,1)
			))

		self.Q = np.eye(self.state_dim)

		self.Ru = self.state_control_weight * np.eye(self.action_dim)

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = -1 * (np.dot(s.T,np.dot(self.Q,s)) + np.dot(a.T,np.dot(self.Ru,a)))
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		reward = np.clip(reward,self.r_min,self.r_max)
		return (reward - self.r_min) / (self.r_max - self.r_min)

	def step(self,s,a,dt):
		Fd = np.eye(self.state_dim) + dt * self.Fc
		Bd = dt * self.Bc
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

	def plot_value_dataset(self,dataset,title):
		
		encodings = dataset[0]
		target = dataset[1] 
		
		# contour
		if encodings.shape[0] > 100:
			fig,ax = plt.subplots(nrows=1,ncols=self.num_robots,squeeze=False)
			state_idx_per_robot = int(self.state_dim / self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot]
				pos_i_idxs = robot_state_idxs * robot + np.arange(robot_state_idxs)[self.position_idx]
				pcm = ax[0,robot].tricontourf(encodings[:,pos_i_idxs[0]],encodings[:,pos_i_idxs[1]],target[:,robot])
				fig.colorbar(pcm,ax=ax[0,robot])
				ax[0,robot].set_title("{} Value for Robot {}".format(title,robot))
				ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
				ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
				self.render(fig=fig,ax=ax[0,robot])

		# scatter
		fig,ax = plt.subplots(nrows=1,ncols=self.num_robots,squeeze=False)
		state_idx_per_robot = int(self.state_dim / self.num_robots)
		for robot in range(self.num_robots):
			robot_state_idxs = self.state_idxs[robot]
			pos_i_idxs = robot_state_idxs * robot + np.arange(robot_state_idxs)[self.position_idx]
			pcm = ax[0,robot].scatter(encodings[:,pos_i_idxs[0]],encodings[:,pos_i_idxs[1]],c=target[:,robot])
			fig.colorbar(pcm,ax=ax[0,robot])
			ax[0,robot].set_title("{} Value for Robot {}".format(title,robot))
			ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
			ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
			self.render(fig=fig,ax=ax[0,robot])


	def plot_policy_dataset(self,dataset,title,robot):

		encodings = dataset[0]
		target = dataset[1] 		

		# quiver plot 
		fig,ax = plt.subplots(nrows=1,ncols=self.num_robots,squeeze=False)
		state_idx_per_robot = int(self.state_dim / self.num_robots)
		robot_state_idxs = self.state_idxs[robot]
		pos_i_idxs = robot_state_idxs * robot + np.arange(robot_state_idxs)[self.position_idx]
		C = np.linalg.norm(target[:,0:1],axis=1)
		ax[0,robot].quiver(encodings[:,pos_i_idxs[0]],encodings[:,pos_i_idxs[1]],\
			target[:,0],target[:,1])
		ax[0,robot].scatter(encodings[:,pos_i_idxs[0]],encodings[:,pos_i_idxs[1]],c=C,s=2)
		ax[0,robot].set_title("{} Policy for Robot {}".format(title,robot))
		ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
		ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
		self.render(fig=fig,ax=ax[0,robot])
