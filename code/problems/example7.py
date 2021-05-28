


# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator , multi robot uncooperative target  
class Example7(Problem):

	def __init__(self): 
		super(Example7,self).__init__()

		self.t0 = 0
		self.tf = 1
		self.dt = 0.1
		self.gamma = 0.99
		self.desired_distance = 1.0
		self.num_robots = 2 
		self.state_dim = 4 
		self.action_dim = 4 
		self.r_max = 100
		self.r_min = -1 * self.r_max 
		self.name = "example7"
		self.position_idx = np.arange(2) 
		self.state_control_weight = 1e-5 

		self.state_idxs = [np.arange(2), 2 + np.arange(2)]
		self.action_idxs = [np.arange(2), 2 + np.arange(2)]
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			(-2,2), 
			(-2,2), 
			# (-5,5), 
			(-2,2), 
			(-2,2),
			# (-5,5),
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

		self.action_lims = 0.5*np.array((
			(-1,1),
			(-1,1),
			# (-1,1),
			# (-1,1),
			(0,0),
			(0,0),
			))

		self.init_lims = np.array((
			(-2,2), 
			(-2,2), 
			# (-5,5), 
			(-2,2), 
			(-2,2),
			# (-5,5),
			))

		self.Fc = np.array((
			(0,0), 
			(0,0), 
			))

		self.Bc = np.array((
			(1,0),
			(0,1),
			))

		self.Q = np.eye(2)
		self.Ru = self.state_control_weight * np.eye(2)

	def reward(self,s,a):
		s_1 = s[self.state_idxs[0]]
		s_2 = s[self.state_idxs[1]]
		a_1 = a[self.action_idxs[1]]
		r = -1 * (
			np.abs((s_1-s_2).T @ self.Q @ (s_1 - s_2) - self.desired_distance) + \
			a_1.T @ self.Ru @ a_1).squeeze()
		reward = np.array([[r],[-r]])
		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		reward = np.clip(reward,self.r_min,self.r_max)
		reward = (reward - self.r_min) / (self.r_max - self.r_min)
		reward = np.array([[reward[0,0]],[1-reward[0,0]]])
		return reward
		
	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		for robot in range(self.num_robots):
			Fd = np.eye(len(self.state_idxs[robot])) +  dt * self.Fc 
			Bd = dt * self.Bc 
			s_tp1[self.state_idxs[robot],:] = np.dot(Fd,s[self.state_idxs[robot],:]) + np.dot(Bd,a[self.action_idxs[robot],:])
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot] 

				ax.plot(states[:,robot_state_idxs[0]], states[:,robot_state_idxs[1]], color=colors[robot])
				ax.plot(states[0,robot_state_idxs[0]], states[0,robot_state_idxs[1]], color=colors[robot],marker='o')
				ax.plot(states[-1,robot_state_idxs[0]], states[-1,robot_state_idxs[1]], color=colors[robot],marker='s')
				
			# ax.set_aspect(lims[0,1]-lims[0,0] / lims[1,1]-lims[1,0])

			for robot in range(self.num_robots):
				ax.plot(np.nan,np.nan,color=colors[robot],label="Robot {}".format(robot))
			ax.legend(loc='best')

		lims = self.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_aspect( (lims[1,1]-lims[1,0]) / (lims[0,1]-lims[0,0]) )

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

		max_plots_per_robot = 10

		for robot in range(self.num_robots):
			groups = self.make_groups(dataset[0],dataset[1],robot)
			if len(groups) > max_plots_per_robot:
				groups = groups[0:max_plots_per_robot]

			for group in groups: 

				data = np.array([np.array(xi) for xi in group])
				encodings = data[:,0:self.state_dim]
				target = data[:,self.state_dim:]

				# contour
				if encodings.shape[0] > 100:
					fig,ax = plt.subplots() 
					robot_idxs = self.state_idxs[robot]
					pcm = ax[0,robot].tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],target[:,robot])
					fig.colorbar(pcm,ax=ax[0,robot])
					ax.set_title("{} Value for Robot {}".format(title,robot))
					ax.set_xlim(self.state_lims[self.position_idx[0],:])
					ax.set_ylim(self.state_lims[self.position_idx[0],:])

				else:
					# scatter
					fig,ax = plt.subplots() 
					robot_idxs = self.state_idxs[robot]
					pcm = ax[0,robot].tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],target[:,robot])
					fig.colorbar(pcm,ax=ax)
					ax.set_title("{} Value for Robot {}".format(title,robot))
					ax.set_xlim(self.state_lims[self.position_idx[0],:])
					ax.set_ylim(self.state_lims[self.position_idx[0],:])

				# plot other agents 
				state = group[0][0:self.state_dim]
				for other_robot in range(self.num_robots):
					if other_robot != robot:
						other_robot_idxs = self.state_idxs[other_robot] 
						# Circle((x,y),radius)
						circ = patches.Circle((state[other_robot_idxs[0]], state[other_robot_idxs[1]]), \
							self.approx_dist,facecolor='gray',alpha=0.5)
						ax.add_patch(circ)
			
				self.render(fig=fig,ax=ax)

	def plot_policy_dataset(self,dataset,title,robot):

		max_plots = 10

		robot_idxs = self.state_idxs[robot]

		groups = self.make_groups(dataset[0],dataset[1],robot)
		if len(groups) > max_plots:
			groups = groups[0:max_plots]

		for group in groups: 
			fig,ax = plt.subplots() 

			data = np.array([np.array(xi) for xi in group])
			encodings = data[:,0:self.state_dim]
			target = data[:,self.state_dim:]

			# quiver 
			C = np.linalg.norm(target[:,0:1],axis=1)
			ax.quiver(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],\
				target[:,0],target[:,1])
			ax.scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=C,s=2)
			ax.set_title("{} Policy for Robot {}".format(title,robot))
			ax.set_xlim(self.state_lims[self.position_idx[0],:])
			ax.set_ylim(self.state_lims[self.position_idx[0],:])

			# plot other agents 
			state = group[0][0:self.state_dim]
			for other_robot in range(self.num_robots):
				if other_robot != robot:
					other_robot_idxs = self.state_idxs[other_robot] 
					# Circle((x,y),radius)
					circ = patches.Circle((state[other_robot_idxs[0]], state[other_robot_idxs[1]]), \
						self.approx_dist,facecolor='gray',alpha=0.5)
					ax.add_patch(circ)
			self.render(fig=fig,ax=ax)


	def make_groups(self,encoding,target,robot):

		num_datapoints = encoding.shape[0]
		groups = [] # list of list of lists
		robot_idxs = self.state_idxs[robot]
		not_robot_idxs = []
		for i in range(self.state_dim):
			if i not in robot_idxs:
				not_robot_idxs.append(i)

		# print('encoding.shape',encoding.shape)
		# print('target.shape',target.shape)

		for i in range(num_datapoints): 
			matched = False
			for group in groups: 
				# if self.isApprox(encoding[i][not_robot_idxs],group[0][0][not_robot_idxs]):
				if self.isApprox(encoding[i][not_robot_idxs],group[0][not_robot_idxs]):
					# group.append([encoding[i].tolist(),target[i].tolist()])
					group.append(np.concatenate((encoding[i],target[i])))
					matched = True
					break 
			if not matched: 
				# groups.append([encoding[i].tolist(),target[i].tolist()])
				groups.append([np.concatenate((encoding[i],target[i]))])
		return groups 

	def isApprox(self,s1,s2):
		return np.linalg.norm(s1-s2) < self.approx_dist 


	def pretty_plot(self,sim_result):

		for robot in [0,1]:

			fig,ax = plt.subplots()
			inital_state = sim_result["states"][0]

			robot_idxs = self.state_idxs[robot]
			not_robot_idxs = []
			for i in range(self.state_dim):
				if i not in robot_idxs:
					not_robot_idxs.append(i)

			num_eval = 3000
			states = []
			for _ in range(num_eval):
				state = self.initialize()
				state[not_robot_idxs,:] = inital_state[not_robot_idxs,:]
				states.append(state)
			states = np.array(states).squeeze(axis=2)

			# plot value func contours
			if sim_result["instance"]["solver"].value_oracle is not None:

				value_oracle = sim_result["instance"]["solver"].value_oracle
				values = []
				for state in states: 
					value = value_oracle.eval(self,state)
					values.append(value)
				values = np.array(values).squeeze(axis=2)

				pcm = ax.tricontourf(states[:,robot_idxs[0]],states[:,robot_idxs[1]],values[:,robot])
				fig.colorbar(pcm,ax=ax)	

			# plot policy function 
			if sim_result["instance"]["solver"].policy_oracle is not [None for _ in range(self.num_robots)]:

				policy_oracle = sim_result["instance"]["solver"].policy_oracle
				actions = []
				for state in states: 
					action = policy_oracle[robot].eval(self,state,robot)
					actions.append(action)
				actions = np.array(actions).squeeze(axis=2)

				ax.quiver(states[:,robot_idxs[0]],states[:,robot_idxs[1]],actions[:,0],actions[:,1])

			
			# plot final trajectory , obstacles and limits 
			self.render(fig=fig,ax=ax,states=sim_result["states"])
