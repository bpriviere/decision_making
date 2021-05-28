

# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# multi-scale bugtrap 2d single integrator , single robot , with obstacles 
class Example11(Problem):

	def __init__(self): 
		super(Example11,self).__init__()

		self.t0 = 0
		self.tf = 40
		self.dt = 0.5
		self.r_max = 10
		self.r_min = -self.r_max
		self.num_robots = 1
		self.gamma = 0.999
		# self.gamma = 1.0
		self.state_dim = 2
		self.action_dim = 2
		self.state_control_weight = 0.01 # 1.0
		self.name = "example11"
		self.position_idx = np.arange(2)
		self.desired_distance = 0.5
		self.s_0 = np.array([[0],[0]])
		self.s_des = np.array([[4],[0]])

		self.state_idxs = [np.arange(2)]
		self.action_idxs = [np.arange(2)]
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

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

		l_slit = 0.5
		self.obstacles = [
			np.array([
				[1,2],
				[-2,2],
			]),
			np.array([
				[-2,2],
				[1,2],
			]),
			np.array([
				[-2,2],
				[-2,-1],
			]),
			np.array([
				[-2,-1],
				[l_slit/2,2],
			]),
			np.array([
				[-2,-1],
				[-l_slit/2,-2],
			])
		]

		self.Fc = np.zeros((self.state_dim,self.state_dim))
		self.Bc = np.eye(self.state_dim)
		self.Q = np.eye(self.state_dim)
		self.Ru = self.state_control_weight * np.eye(self.action_dim)

	def initialize(self,uniform=False):
		if uniform:
			valid = False
			while not valid:
				state = sample_vector(self.init_lims)
				valid = not self.is_terminal(state)
		else:
			state = self.s_0
		return state

	def reward(self,s,a):
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = -1 * (np.dot((s-self.s_des).T,np.dot(self.Q,(s-self.s_des))) + np.dot(a.T,np.dot(self.Ru,a))).squeeze()
		return reward

	def normalized_reward(self,s,a): 
		# reward = self.reward(s,a)
		# r_max = self.r_max
		# r_min = -r_max
		# reward = np.clip(reward,r_min,r_max)
		# return (reward - r_min) / (r_max - r_min)
		reward = np.zeros((self.num_robots,1))
		reward[0,0] = 0
		if np.linalg.norm(s-self.s_des) < self.desired_distance:
			reward[1,0] = 0
		return reward 

	def step(self,s,a,dt):
		Fd = np.eye(self.state_dim) + self.Fc * dt 
		Bd = self.Bc * dt 
		s_tp1 = np.dot(Fd,s) + np.dot(Bd,a)
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		if fig == None or ax == None: 
			fig,ax = plotter.make_fig()	

		state_lims = self.state_lims
		ax.set_xlim([state_lims[0,0],state_lims[0,1]])
		ax.set_ylim([state_lims[1,0],state_lims[1,1]])
		ax.set_aspect((state_lims[1,1]-state_lims[1,0])/(state_lims[0,1]-state_lims[0,0]))

		if states is not None:
			states = np.array(states)
			ax.plot(states[:,0],states[:,1])
			ax.plot(states[0,0],states[0,1],'o',color='b',label="Start")
			ax.plot(states[-1,0],states[-1,1],'s',color='b',label="Terminal")

		# plot obstacles 
		for obstacle in self.obstacles:
			# Rectangle((x,y),width,height)
			rect = patches.Rectangle((obstacle[0,0], obstacle[1,0]), \
				(obstacle[0,1]-obstacle[0,0]), \
				(obstacle[1,1]-obstacle[1,0]), \
				facecolor='gray')
			ax.add_patch(rect)

		# plot goal
		circ = patches.Circle((self.s_des[0],self.s_des[1]),self.desired_distance,facecolor='green',
			alpha=0.5,label="Goal")
		ax.add_patch(circ)

		ax.legend()

		return fig,ax

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims) and not any(contains(state,obstacle) for obstacle in self.obstacles)

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
			for robot in range(self.num_robots):
				robot_idxs = self.state_idxs[robot]
				pcm = ax[0,robot].tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],target[:,robot])
				fig.colorbar(pcm,ax=ax[0,robot])
				ax[0,robot].set_title("{} Value for Robot {}".format(title,robot))
				ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
				ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
				self.render(fig=fig,ax=ax[0,robot])

		else:
			# scatter
			fig,ax = plt.subplots(nrows=1,ncols=self.num_robots,squeeze=False)
			for robot in range(self.num_robots):
				robot_idxs = self.state_idxs[robot]
				pcm = ax[0,robot].scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=target[:,robot])
				fig.colorbar(pcm,ax=ax[0,robot])
				ax[0,robot].set_title("{} Value for Robot {}".format(title,robot))
				ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
				ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
				self.render(fig=fig,ax=ax[0,robot])


	def plot_policy_dataset(self,dataset,title,robot):

		encodings = dataset[0]
		target = dataset[1] 		

		fig,ax = plt.subplots(nrows=1,ncols=self.num_robots,squeeze=False)
		
		# quiver plot 
		robot_idxs = self.state_idxs[robot] 
		ax[0,robot].quiver(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],\
			target[:,0],target[:,1])
		ax[0,robot].scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],s=2)
		ax[0,robot].set_title("{} Policy for Robot {}".format(title,robot))
		ax[0,robot].set_xlim(self.state_lims[self.position_idx[0],:])
		ax[0,robot].set_ylim(self.state_lims[self.position_idx[0],:])
		
		# uncertainty if evaluated from gaussian neural network 
		if target.shape[1] > 2:
			C = np.linalg.norm(target[:,2:],axis=1)
			if encodings.shape[0] > 100:
				pcm = ax[0,robot].tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],C,alpha=0.3)
				fig.colorbar(pcm,ax=ax[0,robot])
			else:
				ax[0,robot].scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=C)
		
		# render
		self.render(fig=fig,ax=ax[0,robot])


	def pretty_plot(self,sim_result):

		fig,ax = plt.subplots()

		num_eval = 3000
		states = []
		for _ in range(num_eval):
			states.append(self.initialize())
		states = np.array(states).squeeze(axis=2)

		# plot value func contours
		if sim_result["instance"]["value_oracle"] is not None:
			value_oracle = sim_result["instance"]["value_oracle"]
			values = []
			for state in states: 
				value = value_oracle.eval(self,state)
				values.append(value)
			values = np.array(values).squeeze(axis=2)

			pcm = ax.tricontourf(states[:,0],states[:,1],values[:,0])
			fig.colorbar(pcm,ax=ax)	

		# plot policy function 
		if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
			policy_oracle = sim_result["instance"]["policy_oracle"]
			actions = []
			robot = 0 
			for state in states: 
				action = policy_oracle[robot].eval(self,state,robot)
				actions.append(action)
			actions = np.array(actions).squeeze(axis=2)

			ax.quiver(states[:,0],states[:,1],actions[:,0],actions[:,1])

		
		# plot final trajectory , obstacles and limits 
		self.render(fig=fig,ax=ax,states=sim_result["states"])

		