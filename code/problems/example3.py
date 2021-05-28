
# standard 
import numpy as np 
import copy 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 3d dubins vehicle , 2 robots 
class Example3(Problem):

	def __init__(self): 
		super(Example3,self).__init__()

		self.t0 = 0
		self.tf = 20
		self.dt = 0.1
		self.gamma = 1.0
		self.desired_distance = 0.2
		self.state_control_weight = 0.1
		self.g = 1.0 
		self.state_dim = 14
		self.action_dim = 6
		self.num_robots = 2 
		self.position_idx = np.arange(3)
		self.r_max = 100
		self.r_min = -1 * self.r_max
		self.name = "example3"

		state_dim_per_robot = 7 
		action_dim_per_robot = 3		
		self.state_idxs = [
			np.arange(state_dim_per_robot),
			np.arange(state_dim_per_robot) + state_dim_per_robot ]
		self.action_idxs = [
			np.arange(action_dim_per_robot),
			np.arange(action_dim_per_robot) + action_dim_per_robot ]

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		self.state_lims = np.array((
			[-10,10],
			[-10,10],
			[-10,10],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[ 1,2],
			[-10,10],
			[-10,10],
			[-10,10],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[ 1,2],
			))

		self.action_lims = np.array((
			[-np.pi/10,np.pi/10],
			[-np.pi/10,np.pi/10],
			[-0.5,0.5],
			[-np.pi/10,np.pi/10],
			[-np.pi/10,np.pi/10],
			[-0.5,0.5],
			))

		self.init_lims = np.array((
			[-10,10],
			[-10,10],
			[-10,10],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[ 1,2],
			[-10,10],
			[-10,10],
			[-10,10],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[-np.pi,np.pi],
			[ 1,2],
			))

		self.Q = np.array((
			(1,0,0,0,0,0,0),
			(0,1,0,0,0,0,0),
			(0,0,1,0,0,0,0),
			(0,0,0,0,0,0,0),
			(0,0,0,0,0,0,0),
			(0,0,0,0,0,0,0),
			(0,0,0,0,0,0,0),
			))

		self.Ru = self.state_control_weight * np.array((
			(1,0,0),
			(0,1,0),
			(0,0,1),
			))

	def reward(self,s,a):
		s_1 = s[self.state_idxs[0]] 
		s_2 = s[self.state_idxs[1]] 
		a_1 = a[self.action_idxs[0]] 
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
		# s = [x,y,z,psi,gamma,phi,v]
		# a = [gammadot, phidot,vdot]
		sdot = np.zeros(s.shape)
		for robot in range(self.num_robots):
			sdot[self.state_idxs[robot][0],0] = s[self.state_idxs[robot][6],0] * np.cos(s[self.state_idxs[robot][4],0]) * np.sin(s[self.state_idxs[robot][3],0])
			sdot[self.state_idxs[robot][1],0] = s[self.state_idxs[robot][6],0] * np.cos(s[self.state_idxs[robot][4],0]) * np.cos(s[self.state_idxs[robot][3],0])
			sdot[self.state_idxs[robot][2],0] = -s[self.state_idxs[robot][6],0] * np.sin(s[self.state_idxs[robot][4],0]) 
			sdot[self.state_idxs[robot][3],0] = self.g / s[self.state_idxs[robot][6],0] * np.tan(s[self.state_idxs[robot][5],0])
			sdot[self.state_idxs[robot][4],0] = a[self.action_idxs[robot][0],0]
			sdot[self.state_idxs[robot][5],0] = a[self.action_idxs[robot][1],0]
			sdot[self.state_idxs[robot][6],0] = a[self.action_idxs[robot][2],0]
		s_tp1 = s + sdot * dt 

		# wrap angles 
		for robot in range(self.num_robots):
			s_tp1[self.state_idxs[robot][3],0] = s_tp1[self.state_idxs[robot][3],0] % (2*np.pi)
			s_tp1[self.state_idxs[robot][4],0] = s_tp1[self.state_idxs[robot][4],0] % (2*np.pi)
			s_tp1[self.state_idxs[robot][5],0] = s_tp1[self.state_idxs[robot][5],0] % (2*np.pi)

		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		if fig is None and ax is None:
			fig,ax = plotter.make_3d_fig()

		if states is not None:
			lims = self.state_lims
			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot] 

				ax.plot(states[:,robot_state_idxs[0]].squeeze(), states[:,robot_state_idxs[1]].squeeze(),states[:,robot_state_idxs[2]].squeeze(), color=colors[robot])
				ax.plot(states[0,robot_state_idxs[0]], states[0,robot_state_idxs[1]], states[0,robot_state_idxs[2]],color=colors[robot],marker='o')
				ax.plot(states[-1,robot_state_idxs[0]], states[-1,robot_state_idxs[1]], states[-1,robot_state_idxs[2]],color=colors[robot],marker='s')

				# projections 
				ax.plot(lims[0,0]*np.ones(states.shape[0]),states[:,robot_state_idxs[1]].squeeze(),states[:,robot_state_idxs[2]].squeeze(),\
					color=colors[robot],linewidth=1,linestyle="--")
				ax.plot(states[:,robot_state_idxs[0]].squeeze(),lims[1,1]*np.ones(states.shape[0]),states[:,robot_state_idxs[2]].squeeze(),\
					color=colors[robot],linewidth=1,linestyle="--")
				ax.plot(states[:,robot_state_idxs[0]].squeeze(),states[:,robot_state_idxs[1]].squeeze(),lims[2,0]*np.ones(states.shape[0]),\
					color=colors[robot],linewidth=1,linestyle="--")

			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))
			ax.set_zlim((lims[2,0],lims[2,1]))
			for robot in range(self.num_robots):
				ax.scatter(np.nan,np.nan,np.nan,color=colors[robot],label="Robot {}".format(robot))
			ax.legend(loc='best')
		return fig,ax 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		s1 = state[self.state_idxs[0],:]
		s2 = state[self.state_idxs[1],:]
		return s2-s1 

	def value_encoding(self,state):
		s1 = state[self.state_idxs[0],:]
		s2 = state[self.state_idxs[1],:]
		return s2-s1 