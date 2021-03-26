
# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 3d dubins vehicle , 2 robots 
class Example3(Problem):

	def __init__(self,\
		t0 = 0,
		tf = 20,
		dt = 0.1,
		pos_lim = 2,
		vel_lim = 2,
		acc_lim = 1.0,
		rad_lim = 2*np.pi,
		omega_lim = 2*np.pi / 10,
		desired_distance = 0.2, 
		state_control_weight = 0.1,
		g = 3.0, 
		): 
		super(Example3,self).__init__()

		times = np.arange(t0,tf,dt)

		num_robots = 2
		state_dim_per_robot = 7
		action_dim_per_robot = 3
		state_dim = state_dim_per_robot * num_robots
		action_dim = action_dim_per_robot * num_robots
		position_idx = np.arange(3)

		# state and action lim 
		state_lims = np.zeros((state_dim,2))
		action_lims = np.zeros((action_dim,2))
		for i in range(num_robots):
			state_shift = state_dim_per_robot * i 
			action_shift = action_dim_per_robot * i 
			
			state_lims[state_shift + np.arange(0,3),0] = -pos_lim
			state_lims[state_shift + np.arange(0,3),1] =  pos_lim
			state_lims[state_shift + np.arange(3,6),0] = -rad_lim
			state_lims[state_shift + np.arange(3,6),1] =  rad_lim
			state_lims[state_shift + 6,0]   =  0.5*vel_lim
			state_lims[state_shift + 6,1]   =  vel_lim

			action_lims[action_shift + np.arange(0,2),0] = -rad_lim
			action_lims[action_shift + np.arange(0,2),1] =  rad_lim
			action_lims[action_shift+2,0] = -acc_lim
			action_lims[action_shift+2,1] =  acc_lim

		# reward 
		Q = np.zeros((state_dim_per_robot,state_dim_per_robot))
		Q[0,0] = 1
		Q[1,1] = 1
		Q[2,2] = 1
		Ru = state_control_weight*np.eye(action_dim_per_robot)
		self.Q = Q
		self.Ru = Ru

		# other
		self.g = g 
		self.desired_distance = desired_distance

		# standard 
		self.num_robots = num_robots
		self.gamma = 1
		self.state_dim = state_dim
		self.state_dim_per_robot = state_dim_per_robot
		self.state_lims = state_lims 
		self.action_dim = action_dim
		self.action_dim_per_robot = action_dim_per_robot
		self.action_lims = action_lims 
		self.position_idx = position_idx 
		self.dt = dt
		self.times = times  
		self.policy_encoding_dim = state_dim
		self.value_encoding_dim = state_dim
		self.name = "example3"

	def sample_action(self):
		return sample_vector(self.action_lims)

	def sample_state(self,damp=0.0):
		return sample_vector(self.state_lims,damp=damp)

	def reward(self,s,a):
		s_1 = s[0:self.state_dim_per_robot]
		s_2 = s[self.state_dim_per_robot:]
		a_1 = a[0:self.action_dim_per_robot]
		r = -1 * (
			np.abs((s_1-s_2).T @ self.Q @ (s_1 - s_2) - self.desired_distance) + \
			a_1.T @ self.Ru @ a_1)
		return r 

	def normalized_reward(self,s,a): 
		r1 = self.reward(s,a)
		r_max = 100
		r_min = -r_max
		r1 = np.clip(r1,r_min,r_max)
		r1 = (r1 - r_min) / (r_max - r_min)
		return np.array([r1,1-r1]).squeeze()

	def step(self,s,a):
		# s = [x,y,z,psi,gamma,phi,v]
		# a = [gammadot, phidot,vdot]
		sdot = np.zeros(s.shape)
		for robot in range(self.num_robots):
			state_shift = robot * self.state_dim_per_robot
			action_shift = robot * self.action_dim_per_robot
			sdot[state_shift+0,0] = s[state_shift+6,0] * np.cos(s[state_shift+4,0]) * np.sin(s[state_shift+3,0])
			sdot[state_shift+1,0] = s[state_shift+6,0] * np.cos(s[state_shift+4,0]) * np.cos(s[state_shift+3,0])
			sdot[state_shift+2,0] = -s[state_shift+6,0] * np.sin(s[state_shift+4,0]) 
			sdot[state_shift+3,0] = self.g / s[state_shift+6,0] * np.tan(s[state_shift+5,0])
			sdot[state_shift+4,0] = a[action_shift+0,0]
			sdot[state_shift+5,0] = a[action_shift+1,0]
			sdot[state_shift+6,0] = a[action_shift+2,0]
		s_tp1 = s + sdot * self.dt 

		# wrap angles 
		for robot in range(self.num_robots):
			state_shift = robot * self.state_dim_per_robot
			s_tp1[state_shift+3,0] = s_tp1[state_shift+3,0] % (2*np.pi)
			s_tp1[state_shift+4,0] = s_tp1[state_shift+4,0] % (2*np.pi)
			s_tp1[state_shift+5,0] = s_tp1[state_shift+5,0] % (2*np.pi)

		return s_tp1 

	def render(self,states):
		# states, np array in [nt x state_dim]
		fig,ax = plotter.make_3d_fig()
		colors = plotter.get_n_colors(self.num_robots)
		for robot in range(self.num_robots):
			state_idxs = robot * self.state_dim_per_robot + np.arange(self.state_dim_per_robot)
			ax.plot(states[:,state_idxs[0]].squeeze(), states[:,state_idxs[1]].squeeze(), \
				states[:,state_idxs[2]].squeeze(), color=colors[robot])
			ax.scatter(states[-1,state_idxs[0]], states[-1,state_idxs[1]], states[-1,state_idxs[2]], color=colors[robot])
		lims = self.state_lims
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


	def initialize(self):
		# return self.sample_state(damp=0.2)
		return self.sample_state()


	def steer(self,s1,s2):
		num_samples = 10 
		for i in range(num_samples):
			action = self.sample_action()
			dist = self.dist(self.step(s1,action),s2)
			if i == 0 or dist < best_dist:
				best_action = action
				best_dist = dist 
		return best_action 

	def dist(self,s1,s2):
		return np.linalg.norm(s1-s2)

	def policy_encoding(self,state,robot):
		s1 = state[np.arange(state_dim_per_robot),:]
		s2 = state[state_dim_per_robot + np.arange(state_dim_per_robot),:]
		return s2-s1 

	def value_encoding(self,state):
		s1 = state[np.arange(state_dim_per_robot),:]
		s2 = state[state_dim_per_robot + np.arange(state_dim_per_robot),:]
		return s2-s1 