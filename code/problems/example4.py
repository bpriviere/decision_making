


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
		self.desired_distance = 2.0 # 0.5
		self.mass = 1
		self.num_robots = 2 
		self.state_dim = 12
		self.action_dim = 6
		self.r_max = 1000
		self.r_min = -1 * self.r_max
		self.name = "example4"
		self.position_idx = np.arange(3) 
		self.state_control_weight = 1e-5 

		state_dim_per_robot = 6 
		action_dim_per_robot = 3 
		self.state_idxs = [np.arange(state_dim_per_robot),state_dim_per_robot+np.arange(state_dim_per_robot)]
		self.action_idxs = [np.arange(action_dim_per_robot),action_dim_per_robot+np.arange(action_dim_per_robot)]

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = self.state_dim
		self.value_encoding_dim = self.state_dim

		# x, y, z, vx, vy, vz
		state_box = np.array((
			(-2,2), 
			(-5,20), 
			(-2,2), 
			(-1,1), 
			(-1,1), 
			(-1,1)
		)) 

		# x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2
		self.state_lims = np.zeros((self.num_robots*state_dim_per_robot,2))
		self.state_lims[0:state_dim_per_robot,:] = state_box
		self.state_lims[state_dim_per_robot:,:] = state_box

		self.init_lims = np.array((
			(-1,1),
			(-4,-4), 
			(-1,1),
			( 0,0),
			# ( 0,0), #( 0.5,0.5),
			( 0.5,0.5),
			( 0,0),
			(-1,1),
			( 0,0),
			(-1,1),
			( 0,0),
			( 0.5,0.5),
			( 0,0),
			))

		# ax1, ay1, az1, ax2, ay2, az2
		self.action_lims = 0.75*np.array((
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
			(-1,1),
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

		self.Q = np.eye(6)
		self.Ru = self.state_control_weight * np.eye(3)

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
		s_tp1 = np.zeros(s.shape)
		for robot in range(self.num_robots):
			Fd = np.eye(len(self.state_idxs[robot])) +  dt * self.Fc 
			Bd = dt * self.Bc 
			s_tp1[self.state_idxs[robot],:] = np.dot(Fd,s[self.state_idxs[robot],:]) + np.dot(Bd,a[self.action_idxs[robot],:])
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_3d_fig()

		if states is not None:

			lims = self.state_lims
			colors = plotter.get_n_colors(self.num_robots)
			for robot in range(self.num_robots):
				robot_state_idxs = self.state_idxs[robot] 

				ax.plot(states[:,robot_state_idxs[0]].squeeze(axis=1), states[:,robot_state_idxs[1]].squeeze(axis=1),states[:,robot_state_idxs[2]].squeeze(axis=1),color=colors[robot])
				ax.plot(states[0,robot_state_idxs[0]], states[0,robot_state_idxs[1]], states[0,robot_state_idxs[2]], color=colors[robot],marker='o')
				ax.plot(states[-1,robot_state_idxs[0]], states[-1,robot_state_idxs[1]], states[-1,robot_state_idxs[2]], color=colors[robot],marker='s')
				
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
			ax.set_box_aspect((lims[0,1]-lims[0,0], lims[1,1]-lims[1,0], lims[2,1]-lims[2,0]))  

			for robot in range(self.num_robots):
				if robot == 0:
					label = "Pursuer"
				else:
					label = "Evader"
				ax.scatter([np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],color=colors[robot],label=label)

			ax.plot(np.nan,np.nan,marker="o",label="Start",color="black")
			ax.plot(np.nan,np.nan,marker="s",label="End",color="black")
			ax.legend(loc='best')

			# plot desired distance sphere from pursuer, from: https://matplotlib.org/2.0.0/examples/mplot3d/surface3d_demo2.html

			# Make data
			sphere_u = np.linspace(0, 2 * np.pi, 100)
			sphere_v = np.linspace(0, np.pi, 100)
			u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
			sphere_x = states[-1, 0] + self.desired_distance * np.outer(np.cos(sphere_u), np.sin(sphere_v))
			sphere_y = states[-1, 1] + self.desired_distance * np.outer(np.sin(sphere_u), np.sin(sphere_v))
			sphere_z = states[-1, 2] + self.desired_distance * np.outer(np.ones(np.size(sphere_u)), np.cos(sphere_v))
			# Plot the surface
			# ax.plot_surface(sphere_x, sphere_y, sphere_z, color='green', alpha=0.5)
			ax.plot_surface(sphere_x, sphere_y, sphere_z, rstride=1, cstride=1, color='green', alpha=0.5)

		return fig,ax 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state 

	def value_encoding(self,state):
		return state 

	def plot_value_dataset(self, dataset, title):
		pass

	def plot_policy_dataset(self, dataset, title, robot):
		pass