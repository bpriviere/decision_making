

# standard 
import numpy as np 
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# homicidal chauffeur problem - isaacs page 27    
class Example9(Problem):

	def __init__(self): 
		super(Example9,self).__init__()
		# state: [x1,y1,x2,y2,theta2,t]
		# actions: [psi,phi]

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.gamma = 1.0
		self.num_robots = 2 
		self.state_idxs = [
			np.arange(2),
			2 + np.arange(3),
		]
		self.action_idxs = [ 
			np.arange(1),
			1+np.arange(1),
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
		self.R = 2.0

		self.state_dim = 6
		self.action_dim = 2
		self.times = np.arange(self.t0,self.tf,self.dt)

		# self.policy_encoding_dim = self.state_dim
		# self.value_encoding_dim = self.state_dim

		self.policy_encoding_dim = 2
		self.value_encoding_dim = 2


		max_angle = np.pi + self.tf * self.w2 / self.R
		self.state_lims = np.array((
			(-30,30), 
			(-30,30), 
			(-30,30), 
			(-30,30), 
			# (-5,5), 
			# (-5,5), 
			# (-5,5), 
			# (-5,5), 
			# (-np.pi,np.pi), 
			(-max_angle,max_angle), 
			# (-np.inf,np.inf), 
			(0,self.tf)
			))
		self.approx_dist = (self.state_lims[0,1] - self.state_lims[0,0])/10 

		# eps = 0.0001
		self.action_lims = np.array((
			(-np.pi,np.pi),
			# (-eps,eps),
			(-1,1),
			))

		# self.init_lims = self.state_lims 
		self.init_lims = np.array((
			(-5,5), 
			(-5,5), 
			(-5,5), 
			(-5,5),
			(-np.pi,np.pi),
			(0,0),
			))

	def initialize(self):
		valid = False
		u_a = np.pi/4
		l_a = -np.pi/2
		u_r = 3.0 
		l_r = 1.2 * self.desired_distance
		while not valid:
			state = sample_vector(self.init_lims)
			if np.random.uniform() < 0.4 : 
				angle = np.random.uniform() * (u_a - l_a) + l_a
				radius = np.random.uniform() * (u_r - l_r) + l_r
				flip_x = 1 
				if np.random.uniform() < 0.5:
					flip_x = -1 
				state[0,0] = flip_x * radius * np.cos(angle)
				state[1,0] = radius * np.sin(angle)
				state[2,0] = 0 
				state[3,0] = 0 
				state[4,0] = 0 
			valid = not self.is_terminal(state)
		return state

	def reward(self,s,a):
		return self.normalized_reward(s,a)

	def normalized_reward(self,s,a): 
		r1 = 0 
		r2 = 0
		if self.is_captured(s) or s[5,0] > self.tf:
			r1 = s[5,0] / self.tf 
			r2 = 1 - r1 
		if not contains(s[self.state_idxs[0],:],self.state_lims[self.state_idxs[0],:]):
			r1 = -1 
		if not contains(s[self.state_idxs[0],:],self.state_lims[self.state_idxs[0],:]):
			r2 = -1 
		reward = np.array([[r1],[r2]])
		return reward

	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		s_dot = np.zeros(s.shape)
		s_dot[0,0] = self.w1 * np.sin(a[0,0]) # m/s 
		s_dot[1,0] = self.w1 * np.cos(a[0,0])
		s_dot[2,0] = self.w2 * np.sin(s[4,0])
		s_dot[3,0] = self.w2 * np.cos(s[4,0])
		s_dot[4,0] = self.w2 / self.R * a[1,0]
		s_dot[5,0] = 1.0
		s_tp1 = s + s_dot * dt 
		return s_tp1 

	def render(self,states=None,actions=None,fig=None,ax=None):
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

				if robot == 1 and actions is not None:
					robot_action_idxs = self.action_idxs[robot]
					ax.quiver(states[:,robot_idxs[0]],states[:,robot_idxs[1]],\
						np.sin(actions[:,robot_action_idxs[0]]),\
						np.cos(actions[:,robot_action_idxs[0]]))
				
			for robot in range(self.num_robots):
				if robot == 0:
					label = "Evader"
				elif robot == 1:
					label = "Pursuer"
				ax.plot(np.nan,np.nan,color=colors[robot],label=label)
			ax.legend(loc='best')

		# lims = self.state_lims
		ax.set_xlim((-15,15))
		ax.set_ylim((-15,15))
		ax.set_aspect( 1 )
		# ax.axis("equal")
		# x0,x1 = ax.get_xlim()
		# y0,y1 = ax.get_ylim()
		# ax.set_aspect(abs(x1-x0)/abs(y1-y0))

		return fig,ax 

	def is_terminal(self,state):
		capture = self.is_captured(state)
		valid = self.is_valid(state)
		return (not valid) or capture

	def is_captured(self,state):
		return np.linalg.norm(state[0:2,0] - state[2:4,0]) < self.desired_distance

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		# state in 6x1
		# new_state in 2x1
		# return state 
		new_state = np.copy(state).squeeze(axis=1)
		new_state = np.expand_dims(new_state,axis=0)
		new_state = self.isaacs_transformation(new_state)
		new_state = new_state.squeeze(axis=0)
		new_state = np.expand_dims(new_state,axis=1)
		return new_state

	def value_encoding(self,state):
		# return state 
		new_state = np.copy(state).squeeze(axis=1)
		new_state = np.expand_dims(new_state,axis=0)
		new_state = self.isaacs_transformation(new_state)
		new_state = new_state.squeeze(axis=0)
		new_state = np.expand_dims(new_state,axis=1)
		return new_state

	def plot_value_dataset(self,dataset,title):

		states = dataset[0] # in [num datapoints x 2] 
		values = dataset[1] # in [num_datapoints x 2] 

		# get data 
		# new_state = self.isaacs_transformation(states) 
		new_state = states

		# new_state = states

		for robot in range(self.num_robots):
			fig,ax = plt.subplots()
			pcm = ax.tricontourf(new_state[:,0],new_state[:,1],values[:,robot])
			fig.colorbar(pcm,ax=ax)
			ax.set_title("{} Value for Robot {}".format(title,robot))
			# self.render(fig=fig,ax=ax)
			self.render_isaacs(fig=fig,ax=ax)

	def plot_policy_dataset(self,dataset,title,robot):

		robot_action_dim = len(self.action_idxs[robot])
		num_datapoints = dataset[0].shape[0]
		
		states = dataset[0]
		if title == "Eval":
			mu = dataset[1][:,0:robot_action_dim]
			logvar = dataset[1][:,robot_action_dim:]
			eps = np.random.randn(num_datapoints,robot_action_dim)
			actions = mu + eps * np.sqrt(np.exp(logvar))
		else:
			actions = dataset[1]
		actions = actions.squeeze()
		
		# next_states = []
		# for ii in range(num_datapoints):
		# 	state = np.expand_dims(states[ii,:],axis=1)
		# 	action = np.zeros((self.action_dim,1))
		# 	action[self.action_idxs[robot],:] = actions[ii]
		# 	next_state = self.step(state,action,self.dt)
		# 	next_states.append(next_state)
		# next_states = np.array(next_states).squeeze(axis=2)

		# get data 
		# new_state  = self.isaacs_transformation(states) # in [num datapoints x 2] 
		# new_next_states = self.isaacs_transformation(next_states)
		# diff = new_next_states-new_state

		# new_state = self.isaacs_transformation(states)
		new_state = states
		# actions = actions.squeeze()

		diff = np.zeros((num_datapoints,2)) 
		diff[:,0] = np.sin(actions)
		diff[:,1] = np.cos(actions)

		# plot quiver 
		fig,ax = plt.subplots()
		ax.quiver(new_state[:,0],new_state[:,1],diff[:,0],diff[:,1])
		ax.set_title("{} Policy for Robot {}".format(title,robot))
		
		if title == "Eval":
			variance = dataset[1][:,robot_action_dim:]
			pcm = ax.tricontourf(new_state[:,0],new_state[:,1],np.linalg.norm(variance,axis=1),alpha=0.3)
			fig.colorbar(pcm,ax=ax)
		self.render_isaacs(fig=fig,ax=ax)

		
	def isaacs_transformation(self,states):
		# states in [num datapoints x 5] 

		new_states = np.zeros((states.shape[0],2))

		# helper
		def rot(th):
			r = np.array([
				[ np.cos(th),-np.sin(th)],
				[ np.sin(th), np.cos(th)],
				])
			return r

		th = states[:,4]

		# transform state for planar representation
		# 	- shift 
		new_states[:,0] = states[:,0] - states[:,2]
		new_states[:,1] = states[:,1] - states[:,3]
		
		# 	- rotate 
		a = np.expand_dims(new_states,axis=2) # in [num datapoints x 2 x 1]
		b = np.transpose(rot(th),(2,0,1)) # in [num datapoints x 2 x 2]
		new_states = np.matmul(b,a).squeeze(axis=2) # in [num_datapoints x 2] 	

		return new_states 


	def dbg_transform_plot(self,states,actions):
		fig,ax = plt.subplots(ncols=2,nrows=1,squeeze=False)
		self.render(states,fig=fig,ax=ax[0,0])
		self.render_isaacs(states=states,actions=actions,fig=fig,ax=ax[0,1])


	def render_isaacs(self,states=None,actions=None,fig=None,ax=None):
		if fig is None and ax is None:
			fig,ax = plt.subplots()
		if states is not None:
			new_states = self.isaacs_transformation(states)
			ax.plot(new_states[:,0],new_states[:,1])
		# lims = self.state_lims
		# ax.set_xlim((lims[0,0],lims[0,1]))
		# ax.set_ylim((lims[1,0],lims[1,1]))
		# ax.set_aspect( (lims[1,1]-lims[1,0]) / (lims[0,1]-lims[0,0]))
		# ax.axis('equal')


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

			num_eval = 3000
			states = []
			encodings = [] 
			for _ in range(num_eval):
				state = self.initialize()
				encoding = self.policy_encoding(state,0)
				states.append(state)
				encodings.append(encoding)
			states = np.array(states) # num datapoints x 6 x 1
			encodings = np.array(encodings).squeeze(axis=2) # num datapoints x 2

			# # plot value func contours
			# if sim_result["instance"]["value_oracle"] is not None:
			# 	value_oracle = sim_result["instance"]["value_oracle"]
			# 	values = []
			# 	for state in states: 
			# 		value = value_oracle.eval(self,state)
			# 		values.append(value)
			# 	values = np.array(values).squeeze(axis=2)
			# 	pcm = ax.tricontourf(encodings[:,0],encodings[:,1],values[:,0])
			# 	fig.colorbar(pcm,ax=ax)	

			# plot policy function 
			if not all([a is None for a in sim_result["instance"]["policy_oracle"]]):
				policy_oracle = sim_result["instance"]["policy_oracle"]

				actions = []
				for state in states: 
					action = np.zeros((self.action_dim,1))
					for robot in range(self.num_robots):
						action[self.action_idxs[robot],:] = policy_oracle[robot].eval(self,state,robot)
					actions.append(action)
				actions = np.array(actions)  # num_datapoints x 2 x 1

				next_states = [] 
				for ii in range(num_eval):
					next_state = self.step(states[ii,:],actions[ii,:],self.dt)
					next_states.append(next_state)
				next_states = np.array(next_states)

				next_states = next_states.squeeze(axis=(2)) # num datapoints x 2
				states = states.squeeze(axis=2)

				new_states = self.isaacs_transformation(states)
				new_next_states = self.isaacs_transformation(next_states)
				diff = new_next_states - new_states
				
				# new_states = encodings
				# diff = np.zeros((actions.shape[0],2)) 
				# diff[:,0] = np.sin(actions)
				# diff[:,1] = np.cos(actions)

				ax.quiver(new_states[:,0],new_states[:,1],diff[:,0],diff[:,1])
				
				ax.set_xlim((-15,15))
				ax.set_ylim((-15,15))
				ax.set_aspect( 1 )

				# self.render_isaacs(fig=fig,ax=ax)

			# plot final trajectory , obstacles and limits 
			sim_result_states = sim_result["states"].squeeze()
			self.render_isaacs(fig=fig,ax=ax,states=sim_result_states)

			# capture radius 
			circ = patches.Circle((0,0),self.desired_distance,facecolor='green',alpha=0.5,label="Capture")
			ax.add_patch(circ)
			ax.legend()

			ax.set_xlim((-15,15))
			ax.set_ylim((-15,15))
			ax.set_aspect( 1 )

			# sim_result_new_states = self.isaacs_transformation(sim_result_states)
			# self.render_isaacs(fig=fig,ax=ax,states=sim_result_new_states)
