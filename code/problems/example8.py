


# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 2d single integrator pursuit evasion 1v1   
class Example8(Problem):

	def __init__(self): 
		super(Example8,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.5
		self.gamma = 1.0
		self.num_robots = 2 
		self.state_dim = 4
		self.action_dim = 4
		self.r_max = 1
		self.r_min = 0
		self.name = "example8"
		self.position_idx = np.arange(2) 
		self.state_control_weight = 1e-5 
		self.desired_distance = 1.0

		self.state_idxs = [
			np.arange(2),
			2+np.arange(2)
		]
		self.action_idxs = [
			np.arange(2),
			2+np.arange(2)
		]

		self.state_dim = 5
		self.action_dim = 4
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			(-10,10), 
			(-10,10), 
			(-10,10), 
			(-10,10),
			(0,self.tf),
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

		self.action_lims = np.array((
			(-0.5,0.5),
			(-0.5,0.5),
			# (-0.0,0.0),
			# (-0.0,0.0),
			(-0.5,0.5),
			(-0.5,0.5),
			))

		self.init_lims = np.array((
			(-2,2), 
			(-2,2), 
			(-2,2), 
			(-2,2),
			(0,0),
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
		reward = self.normalized_reward(s,a) 
		return reward

	def normalized_reward(self,s,a):
		r1 = 0 
		r2 = 0
		if self.is_captured(s) or s[4,0] > self.tf:
			r1 = s[4,0] / self.tf 
			r2 = 1 - r1 
		if not contains(s[self.state_idxs[0],:],self.state_lims[self.state_idxs[0],:]):
			r1 = -1 
		if not contains(s[self.state_idxs[1],:],self.state_lims[self.state_idxs[1],:]):
			r2 = -1 
		reward = np.array([[r1],[r2]])
		return reward

	def is_captured(self,s):
		return np.linalg.norm(s[self.state_idxs[0],:]-s[self.state_idxs[1],:]) < self.desired_distance


	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		for robot in range(self.num_robots):
			Fd = np.eye(len(self.state_idxs[robot])) +  dt * self.Fc 
			Bd = dt * self.Bc 
			s_tp1[self.state_idxs[robot],:] = np.dot(Fd,s[self.state_idxs[robot],:]) + np.dot(Bd,a[self.action_idxs[robot],:])
		s_tp1[4,0] = s[4,0] + dt 
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

				if robot == 0:
					circ = patches.Circle((states[-1,0], states[-1,1]), \
						self.desired_distance,facecolor='green',alpha=0.5)
					ax.add_patch(circ)

			for robot in range(self.num_robots):
				if robot == 0:
					label = "Evader"
				elif robot == 1:
					label = "Pursuer"
				ax.plot(np.nan,np.nan,color=colors[robot],label=label)
			ax.legend(loc='best')

		lims = self.state_lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_aspect( (lims[1,1]-lims[1,0]) / (lims[0,1]-lims[0,0]) )

		return fig,ax 

	def is_terminal(self,state):
		# return not self.is_valid(state)
		return (not self.is_valid(state)) or self.is_captured(state)

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
					pcm = ax.tricontourf(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],target[:,robot])
					fig.colorbar(pcm,ax=ax)
					ax.set_title("{} Value for Robot {}".format(title,robot))
					ax.set_xlim(self.state_lims[self.position_idx[0],:])
					ax.set_ylim(self.state_lims[self.position_idx[0],:])

				else:
					# scatter
					fig,ax = plt.subplots() 
					robot_idxs = self.state_idxs[robot]
					pcm = ax.scatter(encodings[:,robot_idxs[0]],encodings[:,robot_idxs[1]],c=target[:,robot])
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

		value_plot_on = sim_result["instance"]["value_oracle"] is not None
		policy_plot_on = not all([a is None for a in sim_result["instance"]["policy_oracle"]])

		if value_plot_on or policy_plot_on:

			for robot in [0,1]:

				fig,ax = plt.subplots()
				inital_state = sim_result["states"][0]

				robot_idxs = robot * self.state_dim_per_robot + np.arange(self.state_dim_per_robot)
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
				if sim_result["instance"]["value_oracle"] is not None:
					value_oracle = sim_result["instance"]["value_oracle"]
					values = []
					for state in states: 
						value = value_oracle.eval(self,state)
						values.append(value)
					values = np.array(values).squeeze(axis=2)

					pcm = ax.tricontourf(states[:,robot_idxs[0]],states[:,robot_idxs[1]],values[:,robot])
					fig.colorbar(pcm,ax=ax)	

				# plot policy function 
				if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
					policy_oracle = sim_result["instance"]["policy_oracle"]
					actions = []
					for state in states: 
						action = policy_oracle[robot].eval(self,state,robot)
						actions.append(action)
					actions = np.array(actions).squeeze(axis=2)

					ax.quiver(states[:,robot_idxs[0]],states[:,robot_idxs[1]],actions[:,0],actions[:,1])

		# for robot in [0,1]:

		# 	fig,ax = plt.subplots()
		# 	inital_state = sim_result["states"][0]

		# 	robot_idxs = self.state_idxs[robot]
		# 	not_robot_idxs = []
		# 	for i in range(self.state_dim):
		# 		if i not in robot_idxs:
		# 			not_robot_idxs.append(i)

		# 	num_eval = 3000
		# 	states = []
		# 	for _ in range(num_eval):
		# 		state = self.initialize()
		# 		state[not_robot_idxs,:] = inital_state[not_robot_idxs,:]
		# 		states.append(state)
		# 	states = np.array(states).squeeze(axis=2)

		# 	# plot value func contours
		# 	if sim_result["instance"]["value_oracle"] is not None:
		# 		value_oracle = sim_result["instance"]["value_oracle"]
		# 		values = []
		# 		for state in states: 
		# 			value = value_oracle.eval(self,state)
		# 			values.append(value)
		# 		values = np.array(values).squeeze(axis=2)

		# 		pcm = ax.tricontourf(states[:,robot_idxs[0]],states[:,robot_idxs[1]],values[:,robot])
		# 		fig.colorbar(pcm,ax=ax)	

		# 	# plot policy function 
		# 	if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
		# 		policy_oracle = sim_result["instance"]["policy_oracle"]
		# 		actions = []
		# 		for state in states: 
		# 			action = policy_oracle[robot].eval(self,state,robot)
		# 			actions.append(action)
		# 		actions = np.array(actions).squeeze(axis=2)

		# 		ax.quiver(states[:,robot_idxs[0]],states[:,robot_idxs[1]],actions[:,0],actions[:,1])
				
		# 	# plot final trajectory , obstacles and limits 
		# 	self.render(fig=fig,ax=ax,states=sim_result["states"])
