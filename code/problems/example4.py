


# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 3d double integrator , multi robot uncooperative target  
class Example4(Problem):

	def __init__(self): 
		super(Example4,self).__init__()

		self.t0 = 0
		self.tf = 10
		self.dt = 0.1
		self.gamma = 1.0
		self.desired_distance = 0.5
		self.mass = 1
		self.num_robots = 2 
		self.state_dim_per_robot = 6 
		self.action_dim_per_robot = 3
		self.r_max = 1000
		self.name = "example4"
		self.position_idx = np.arange(3) 
		self.state_control_weight = 1e-5 

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.state_dim = self.num_robots * self.state_dim_per_robot
		self.action_dim = self.num_robots * self.action_dim_per_robot
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			(-2,2), 
			(-5,5), 
			(-2,2), 
			(-1,1), 
			(-1,1), 
			(-1,1), 
			(-2,2), 
			(-5,5),
			(-2,2),
			(-1,1),
			(-1,1),
			(-1,1),
			))

		self.action_lims = 0.75*np.array((
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
			))

		self.init_lims = np.array((
			(-1,1),
			(-4,-4), 
			(-1,1),
			( 0,0),
			( 0,0), #( 0.5,0.5),
			( 0,0),
			(-1,1),
			( 0,0),
			(-1,1),
			( 0,0),
			( 0,0), #( 0.5,0.5),
			( 0,0),
			))

		self.Fc = np.array((
			(0,0,0,1,0,0),
			(0,0,0,0,1,0),
			(0,0,0,0,0,1),
			(0,0,0,0,0,0),
			(0,0,0,0,0,0),
			(0,0,0,0,0,0),
			))

		self.Bc = self.mass * np.array((
			(0,0,0),
			(0,0,0),
			(0,0,0),
			(1,0,0),
			(0,1,0),
			(0,0,1),
			))

		self.Q = np.eye(self.state_dim_per_robot)
		self.Ru = self.state_control_weight * np.eye(self.action_dim_per_robot)

	def reward(self,s,a):
		s_1 = s[0:self.state_dim_per_robot]
		s_2 = s[self.state_dim_per_robot:]
		a_1 = a[0:self.action_dim_per_robot]
		r = -1 * (
			np.abs((s_1-s_2).T @ self.Q @ (s_1 - s_2) - self.desired_distance) + \
			a_1.T @ self.Ru @ a_1).squeeze()
		reward = np.array([[r],[-r]])
		return reward

	def normalized_reward(self,s,a): 
		r0 = self.reward(s,a)[0,0]
		r_max = self.r_max
		r_min = -r_max
		r0 = np.clip(r0,r_min,r_max)
		r0 = (r0 - r_min) / (r_max - r_min)
		_normalized_reward = np.array([[r0],[1-r0]])
		return _normalized_reward

	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		for robot in range(self.num_robots):
			state_idx = robot * self.state_dim_per_robot + np.arange(self.state_dim_per_robot)
			action_idx = robot * self.action_dim_per_robot + np.arange(self.action_dim_per_robot)
			Fd = np.eye(self.state_dim_per_robot) +  dt * self.Fc 
			Bd = dt * self.Bc 
			s_tp1[state_idx,:] = np.dot(Fd,s[state_idx,:]) + np.dot(Bd,a[action_idx,:])
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_3d_fig()

		if states is not None:

			lims = self.state_lims
			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				state_idxs = robot * self.state_dim_per_robot + np.arange(self.state_dim_per_robot)

				ax.plot(states[:,state_idxs[0]].squeeze(axis=1), states[:,state_idxs[1]].squeeze(axis=1),states[:,state_idxs[2]].squeeze(axis=1),color=colors[robot])
				ax.plot(states[0,state_idxs[0]], states[0,state_idxs[1]], states[0,state_idxs[2]], color=colors[robot],marker='o')
				ax.plot(states[-1,state_idxs[0]], states[-1,state_idxs[1]], states[-1,state_idxs[2]], color=colors[robot],marker='s')
				
				# projections 
				ax.plot(lims[0,0]*np.ones(states.shape[0]),states[:,state_idxs[1]].squeeze(),states[:,state_idxs[2]].squeeze(),\
					color=colors[robot],linewidth=1,linestyle="--")
				ax.plot(states[:,state_idxs[0]].squeeze(),lims[1,1]*np.ones(states.shape[0]),states[:,state_idxs[2]].squeeze(),\
					color=colors[robot],linewidth=1,linestyle="--")
				ax.plot(states[:,state_idxs[0]].squeeze(),states[:,state_idxs[1]].squeeze(),lims[2,0]*np.ones(states.shape[0]),\
					color=colors[robot],linewidth=1,linestyle="--")

			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))
			ax.set_zlim((lims[2,0],lims[2,1]))
			ax.set_box_aspect((lims[0,1]-lims[0,0], lims[1,1]-lims[1,0], lims[2,1]-lims[2,0]))  

			for robot in range(self.num_robots):
				ax.scatter(np.nan,np.nan,np.nan,color=colors[robot],label="Robot {}".format(robot))
			ax.legend(loc='best')

		return fig,ax 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 

