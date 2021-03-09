
# standard 
import numpy as np 

# custom 
from problems.problem import POSG
from problems.types.space import Cube
import plotter 

# 
t0 = 0 
tf = 20
dt = 0.1
times = np.arange(t0,tf,dt)

# 
num_robots = 2
state_dim_per_robot = 7
action_dim_per_robot = 3
state_dim = state_dim_per_robot * num_robots
action_dim = action_dim_per_robot * num_robots
position_idx = np.arange(3)

# reward 
Q = np.zeros((state_dim_per_robot,state_dim_per_robot))
Q[0,0] = 1
Q[1,1] = 1
Q[2,2] = 1
d = 0.2
Ru = 0.1*np.eye(action_dim_per_robot)

# state and action lim 
pos_lim = 2 # m 
rad_lim = np.pi # rad 
vel_lim = 2  # m / s
omega_lim = 2*np.pi / 10 # rad / s 
acc_lim = 1.0 # m / s^2

# other constants
g = 0.98 # m / s^2


class Example2(POSG):

	def __init__(self): 
		S = self.make_S()
		A = self.make_A()
		O = None 
		Z = None 
		R = self.reward
		T = None 
		b0 = None 
		gamma = 1.0 
		I = np.arange(num_robots)
		self.dt = dt
		self.times = times 
		self.position_idx = position_idx
		super(Example2,self).__init__(S,A,O,Z,R,T,b0,gamma,I)


	def is_terminal(self,state):
		return not self.is_valid(state)


	def is_valid(self,state):
		return self.S.contains(state) 


	def initialize(self):
		return self.S.sample(damp=0.2)


	def reward(self,s,a):
		s_1 = s[0:state_dim_per_robot]
		s_2 = s[state_dim_per_robot:]
		a_1 = a[0:action_dim_per_robot]
		r = -1 * (
			np.abs((s_1-s_2).T @ Q @ (s_1 - s_2) - d) + \
			a_1.T @ Ru @ a_1)
		return r 


	def normalized_reward(self,s,a):
		# needs to be in [0,1]
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
		for robot in range(num_robots):
			state_shift = robot * state_dim_per_robot
			action_shift = robot * action_dim_per_robot
			sdot[state_shift+0,0] = s[state_shift+6,0] * np.cos(s[state_shift+4,0]) * np.sin(s[state_shift+3,0])
			sdot[state_shift+1,0] = s[state_shift+6,0] * np.cos(s[state_shift+4,0]) * np.cos(s[state_shift+3,0])
			sdot[state_shift+2,0] = -s[state_shift+6,0] * np.sin(s[state_shift+4,0]) 
			sdot[state_shift+3,0] = g / s[state_shift+6,0] * np.tan(s[state_shift+5,0])
			sdot[state_shift+4,0] = a[action_shift+0,0]
			sdot[state_shift+5,0] = a[action_shift+1,0]
			sdot[state_shift+6,0] = a[action_shift+2,0]
		s_tp1 = s + sdot * self.dt 
		return s_tp1 


	def make_S(self):
		state_lims = np.zeros((state_dim,2))
		for i in range(num_robots):
			shift = i * state_dim_per_robot 
			state_lims[shift+np.arange(3),0] = -pos_lim 
			state_lims[shift+np.arange(3),1] = pos_lim 
			state_lims[shift+3+np.arange(3),0] = -rad_lim 
			state_lims[shift+3+np.arange(3),1] = rad_lim 
			state_lims[shift+6,0] = 0.5*vel_lim
			state_lims[shift+6,1] = vel_lim 
		return Cube(state_lims)


	def make_A(self):
		action_lims = np.zeros((action_dim,2))
		for i in range(num_robots):
			shift = i * action_dim_per_robot 
			action_lims[shift+np.arange(2),0] = -omega_lim
			action_lims[shift+np.arange(2),1] = omega_lim
			action_lims[shift+2,0] = -acc_lim
			action_lims[shift+2,1] = acc_lim
		return Cube(action_lims)


	def render(self,states):
		# states, np array in [nt x state_dim]
		fig,ax = plotter.make_3d_fig()
		colors = plotter.get_n_colors(num_robots)
		for robot in range(num_robots):
			state_idxs = robot * state_dim_per_robot + np.arange(state_dim_per_robot)
			ax.plot(states[:,state_idxs[0]].squeeze(), states[:,state_idxs[1]].squeeze(), states[:,state_idxs[2]].squeeze(), color=colors[robot])
			ax.scatter(states[0,state_idxs[0]], states[0,state_idxs[1]], states[0,state_idxs[2]], color=colors[robot])
			ax.scatter(states[-1,state_idxs[0]], states[-1,state_idxs[1]], states[-1,state_idxs[2]], color=colors[robot])

		lims = self.S.lims
		ax.set_xlim((lims[0,0],lims[0,1]))
		ax.set_ylim((lims[1,0],lims[1,1]))
		ax.set_zlim((lims[2,0],lims[2,1]))

		for robot in range(num_robots):
			ax.scatter(np.nan,np.nan,np.nan,color=colors[robot],label="Robot {}".format(robot))
		ax.legend(loc='best')

		return fig,ax 
