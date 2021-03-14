
# standard 
import numpy as np 

# custom 
from problems.problem import LQR
from problems.types.space import Cube
import plotter 

t0 = 0 
tf = 10
dt = 0.1
times = np.arange(t0,tf,dt)

# dynamics
m = 1
name = "double_integrator"
if name == "single_integrator":
	Fc = np.zeros((state_dim,state_dim)) 
	Bc = np.eye(state_dim)
	position_idx = np.arange(2)
	velocity_idx = [] 
elif name == "double_integrator":
	Fc = np.array(((0,0,1,0),(0,0,0,1),(0,0,0,0),(0,0,0,0))) 
	Bc = 1/m * np.array(((0,0),(0,0),(1,0),(0,1)))
	position_idx = np.arange(2)
	velocity_idx = np.arange(2) + 2
state_dim,action_dim = Bc.shape
F = np.eye(state_dim) +  Fc * dt 
B = Bc * dt
Q = np.eye(state_dim)
Ru = np.eye(action_dim)

# state and action lim 
pos_lim = 10
vel_lim = 5
action_lim = 5


class Example1(LQR):


	def __init__(self): 
		self.dt = dt
		self.times = times 
		self.position_idx = position_idx
		S = self.make_S()
		A = self.make_A()
		super(Example1,self).__init__(F,B,Q,Ru,S,A)

		# learning 
		self.policy_encoding_dim = state_dim
		self.value_encoding_dim = state_dim


	def is_terminal(self,state):
		return not self.is_valid(state)


	def is_valid(self,state):
		return self.S.contains(state)


	def initialize(self):
		return self.S.sample() 
		

	def normalized_reward(self,s,a):
		# needs to be in [0,1]
		reward = self.reward(s,a)
		r_max = 100
		r_min = -r_max
		reward = np.clip(reward,r_min,r_max)
		return (reward - r_min) / (r_max - r_min)


	def make_S(self):
		state_lims = np.zeros((state_dim,2))
		for i_s in position_idx:
			state_lims[i_s,0] = -pos_lim
			state_lims[i_s,1] =  pos_lim 			
		for i_s in velocity_idx:
			state_lims[i_s,0] = -vel_lim
			state_lims[i_s,1] =  vel_lim					
		return Cube(state_lims)


	def make_A(self):
		action_lims = np.zeros((action_dim,2))
		for i_a in range(action_dim):
			action_lims[i_a,0] = -action_lim
			action_lims[i_a,1] =  action_lim
		return Cube(action_lims)		


	def render(self,states):
		states = np.array(states)
		state_lims = self.S.lims

		fig,ax = plotter.make_fig() 
		ax.plot(states[:,0],states[:,1])
		ax.plot(states[0,0],states[0,1],'o')
		ax.plot(states[-1,0],states[-1,1],'s')
		ax.set_xlim([state_lims[0,0],state_lims[0,1]])
		ax.set_ylim([state_lims[1,0],state_lims[1,1]])

	# learning 
	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 