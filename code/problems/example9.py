

# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# homicidal chauffeur problem - isaacs page 27    
class Example9(Problem):

	def __init__(self): 
		super(Example9,self).__init__()
		# state: [x1,y1,x2,y2,theta2]
		# actions: [psi,phi]

		self.t0 = 0
		self.tf = 20
		self.dt = 0.2
		self.gamma = 0.99
		self.num_robots = 2 
		self.state_idxs = [
			np.arange(2),
			2 + np.arange(3)
		]
		self.action_idxs = [ 
			np.arange(1),
			1+np.arange(1)
		]
		self.r_max = 1.0 
		self.r_min = 0.0
		self.name = "example9"
		self.position_idx = np.arange(2) 
		self.state_control_weight = 0.01 
		
		# problem specific parameters 
		self.desired_distance = 1.0
		self.w1 = 1.0 
		self.w2 = 2.0
		self.R = 1.0

		self.action_dim_per_robot = 1 

		self.state_dim = 1 + self.state_idxs[-1][-1] 
		self.action_dim = 1 + self.action_idxs[-1][-1] 
		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			(-10,10), 
			(-10,10), 
			(-10,10), 
			(-10,10), 
			# (-5,5), 
			# (-5,5), 
			# (-5,5), 
			# (-5,5), 
			(-np.pi,np.pi), 
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

		self.action_lims = np.array((
			(-np.pi,np.pi),
			(-1,1),
			))

		# self.init_lims = self.state_lims 
		self.init_lims = np.array((
			(-5,5), 
			(-5,5), 
			(-5,5), 
			(-5,5),
			(-np.pi,np.pi),
			))

	# def initialize(self):
	# 	valid = False
	# 	while not valid:
	# 		state = sample_vector(self.init_lims)
	# 		state[2,0] = 0.0
	# 		state[3,0] = 0.0
	# 		state[4,0] = 0.0
	# 		valid = not self.is_terminal(state)
	# 	return state


	# def reward(self,s,a):
	# 	r = 1 # time until capture reward 
	# 	if self.is_captured(s):
	# 		r = 0.0
	# 	reward = np.array([[r],[-r]])
	# 	return reward

	# def normalized_reward(self,s,a): 
	# 	reward = self.reward(s,a)
	# 	reward = np.clip(reward,self.r_min,self.r_max)
	# 	reward = (reward - self.r_min) / (self.r_max - self.r_min)
	# 	reward = np.array([[reward[0,0]],[1-reward[0,0]]])
	# 	return reward

	def reward(self,s,a):
		return self.normalized_reward(s,a)

	def normalized_reward(self,s,a): 
		# weighted sum of validity of both robots, and then the time to capture
		r1 = contains(s[self.state_idxs[0],:],self.state_lims[self.state_idxs[0],:])
		r2 = contains(s[self.state_idxs[1],:],self.state_lims[self.state_idxs[1],:])
		r3 = 1 
		w1 = 0.1
		w2 = 0.1 
		w3 = 0.8 
		r = w1*r1 + w2*(1-r2) + w3*r3
		reward = np.array([[r],[1-r]])
		return reward

	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		s_dot = np.zeros(s.shape)
		s_dot[0,0] = self.w1 * np.sin(a[0,0])
		s_dot[1,0] = self.w1 * np.cos(a[0,0])
		s_dot[2,0] = self.w2 * np.sin(s[4,0])
		s_dot[3,0] = self.w2 * np.cos(s[4,0])
		s_dot[4,0] = self.w2 / self.R * a[1,0]
		s_tp1 = s + s_dot * dt 

		# wrap angles
		s_tp1[4,0] = ((s_tp1[4,0] + np.pi) % (2*np.pi)) - np.pi 

		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_fig()

		if states is not None:

			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):

				robot_idxs = self.state_idxs[robot]

				ax.plot(states[:,robot_idxs[0]], states[:,robot_idxs[1]], color=colors[robot])
				ax.plot(states[0,robot_idxs[0]], states[0,robot_idxs[1]], color=colors[robot],marker='o')
				ax.plot(states[-1,robot_idxs[0]], states[-1,robot_idxs[1]], color=colors[robot],marker='s')

				if robot == 0:
					circ = patches.Circle((states[-1,robot_idxs[0]], states[-1,robot_idxs[1]]), \
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
		capture = self.is_captured(state)
		valid = self.is_valid(state)
		return (not valid) or capture
		# valid = self.is_valid(state)
		# return not valid 

	def is_captured(self,state):
		return np.linalg.norm(state[0:2,0] - state[2:4,0]) < self.desired_distance

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 

	def plot_value_dataset(self,dataset,title):

		# get data 
		new_state = self.isaacs_transformation(dataset[0]) # in [num datapoints x 2] 
		target = dataset[1] # in [num_datapoints x 2] 

		for robot in range(self.num_robots):
			fig,ax = plt.subplots()
			pcm = ax.tricontourf(new_state[:,0],new_state[:,1],target[:,robot])
			fig.colorbar(pcm,ax=ax)
			ax.set_title("{} Value for Robot {}".format(title,robot))
			self.render(fig=fig,ax=ax)

	def plot_policy_dataset(self,dataset,title,robot):

		# get data 
		new_state = self.isaacs_transformation(dataset[0]) # in [num datapoints x 2] 
		target = dataset[1] # in [num_datapoints x 2] 

		fig,ax = plt.subplots()
		ax.quiver(new_state[:,0],new_state[:,1],np.sin(target[:,0]),np.cos(target[:,0]))
		ax.set_title("{} Policy for Robot {}".format(title,robot))
		if title == "Eval":
			pcm = ax.tricontourf(new_state[:,0],new_state[:,1],target[:,1],alpha=0.3)
			fig.colorbar(pcm,ax=ax)
		self.render(fig=fig,ax=ax)

		
	def isaacs_transformation(self,states):
		# states in [num datapoints x 5] 

		# helper
		def rot(th):
			gamma = th - np.pi/2 
			# r = np.array([
			# 	[np.cos(gamma), -np.sin(gamma)],
			# 	[np.sin(gamma), np.cos(gamma)],
			# 	])
			r = np.array([
				[np.ones(th.shape), np.zeros(th.shape)],
				[np.zeros(th.shape), np.ones(th.shape)],
				])
			return r

		# transform state for planar representation
		# 	- shift 
		ref = np.zeros(states.shape)
		ref[:,2] = states[:,2] 
		ref[:,3] = states[:,3] 
		states = states - ref 
		# 	- rotate 
		a = np.expand_dims(states[:,0:2],axis=2) # in [num datapoints x 2 x 1]
		b = np.transpose(rot(states[:,4]),(2,0,1)) # in [num datapoints x 2 x 2]
		new_states = np.matmul(b,a).squeeze(axis=2) # in [num_datapoints x 2] 	
		# new_states = a	
		return new_states 


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

			fig,ax = plt.subplots()

			# inital_state = sim_result["states"][0]

			# robot_idxs = self.state_idxs[robot] 
			# not_robot_idxs = []
			# for i in range(self.state_dim):
			# 	if i not in robot_idxs:
			# 		not_robot_idxs.append(i)

			# num_eval = 3000
			# states = []
			# for _ in range(num_eval):
			# 	state = self.initialize()
			# 	state[not_robot_idxs,:] = inital_state[not_robot_idxs,:]
			# 	states.append(state)
			# states = np.array(states).squeeze(axis=2)

			num_eval = 3000
			states = []
			for _ in range(num_eval):
				state = self.initialize()
				state[2,0] = 0
				state[3,0] = 0
				state[4,0] = 0
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

				pcm = ax.tricontourf(states[:,0],states[:,1],values[:,0])
				fig.colorbar(pcm,ax=ax)	

			# # plot policy function 
			# if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
			# 	policy_oracle = sim_result["instance"]["policy_oracle"]
			# 	actions = []
			# 	for state in states: 
			# 		action = policy_oracle[robot].eval(self,state,robot)
			# 		actions.append(action)
			# 	actions = np.array(actions).squeeze(axis=2)

			# 	ax.quiver(states[:,robot_idxs[0]],states[:,robot_idxs[1]],np.sin(actions[:,0]),np.cos(actions[:,0]))

				
				# plot final trajectory , obstacles and limits 
				# self.render(fig=fig,ax=ax,states=sim_result["states"])
