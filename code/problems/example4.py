


# standard 
import numpy as np 

# custom 
from problems.problem import Problem
from util import sample_vector, contains
import plotter 

# 3d double integrator , multi robot uncooperative target  
# robot 0 is pursuer 
# robot 1 is evader
class Example4(Problem):

	def __init__(self): 
		super(Example4,self).__init__()

		self.t0 = 0
		self.tf = 50
		self.dt = 0.5
		self.gamma = 1.0
		self.mass = 1
		self.num_robots = 2 
		self.state_dim = 12
		self.action_dim = 6
		self.name = "example4"
		self.position_idx = np.arange(3) 

		# unused parameters 
		self.desired_distance = 10.0 # 0.5
		self.normalized_desired_distance = 0.10
		self.r_max = 1000
		self.r_min = -1 * self.r_max
		self.state_control_weight = 1e-5 


		state_dim_per_robot = 6 
		action_dim_per_robot = 3 
		self.state_idxs = [np.arange(state_dim_per_robot),state_dim_per_robot+np.arange(state_dim_per_robot)]
		self.action_idxs = [np.arange(action_dim_per_robot),action_dim_per_robot+np.arange(action_dim_per_robot)]

		self.times = np.arange(self.t0,self.tf,self.dt)
		self.policy_encoding_dim = 6
		self.value_encoding_dim = 6

		# x, y, z, vx, vy, vz
		self.state_lims = np.array((
			# # 7/30
			# (-1000,1000), 
			# (-1000,1000), 
			# (-1000,1000), 
			# (-15,15), 
			# (-15,15), 
			# (-15,15),
			# (-1000,1000), 
			# (-1000,1000), 
			# (-1000,1000), 
			# (-15,15), 
			# (-15,15), 
			# (-15,15),
			# 8/22
			(-1000,1000), 
			(-1000,1000), 
			(-1000,1000), 
			(-11,11), 
			(-11,11), 
			(-11,11),
			(-1000,1000), 
			(-1000,1000), 
			(-1000,1000), 
			(-11,11), 
			(-11,11), 
			(-11,11),
		)) 

		self.init_lims = np.array((
			# 08/10
			# (-25,25), 
			# (-25,25), 
			# (-25,25), 
			# (-1.0,1.0), 
			# (-1.0,1.0), 
			# (-1.0,1.0),
			# (-25,25), 
			# (-25,25), 
			# (-25,25), 
			# (-1.0,1.0), 
			# (-1.0,1.0), 
			# (-1.0,1.0),
			# 8/22
			(0,0), 
			(0,0), 
			(0,0), 
			(-11,11), 
			(-11,11), 
			(-11,11),
			(-100, 100), # (loop in init until in elevation cone)
			(-100, 100),
			(-100, 100),
			(-11,11), 
			(-11,11), 
			(-11,11),
			))

		# ax1, ay1, az1, ax2, ay2, az2
		self.action_lims = np.array((
			# 07/30
			# (-2,2),
			# (-2,2),
			# (-2,2),
			# (-2,2),
			# (-2,2),
			# (-2,2),
			# 08/22
			(-3,3),
			(-3,3),
			(-3,3),
			(-2,2),
			(-2,2),
			(-2,2),
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

	# overwrite
	def initialize(self):
		valid = False
		while not valid:
			state = sample_vector(self.init_lims)

			# apply min speed bound
			state = self.apply_min_speed(state)

			# align relative velocities 
			v2 = state[9:12,0]
			v1 = v2 + 2 * np.random.normal(size=(3,))
			state[3:6,0] = v1

			valid = not self.is_terminal(state) and self.in_elevation_cone(state)
		return state

	def apply_min_speed(self, state):
		min_speed = 7 
		v2 = state[9:12,0] 
		ratio = np.linalg.norm(v2) / min_speed 
		if ratio < 1:
			state[9,0] = state[9,0] / ratio
			state[10,0] = state[10,0] / ratio
			state[11,0] = state[11,0] / ratio
		return state

	def in_elevation_cone(self, state):
		cone_angle = 35 * np.pi / 180
		dxy = np.linalg.norm(state[0:2,0] - state[6:8,0])
		dz = np.abs(state[2,0] - state[8,0])
		rel_angle = np.arctan2(dz, dxy)
		if rel_angle < cone_angle:
			return True
		else:
			return False

	def reward(self,s,a):
		s_1 = s[self.state_idxs[0]] # n x 1
		s_2 = s[self.state_idxs[1]]
		a_1 = a[self.action_idxs[0]]

		# new
		Q = np.zeros((6,6))
		for i in range(3):
			Q[i,i] = 1 / (100) ** 2.0

		x = (s_1 - s_2).T @ Q @ (s_1 - s_2)

		reward = np.zeros((2,1))
		reward[0,0] = 1 - x 
		reward[1,0] = x

		reward[0,0] = np.min((np.max((reward[0,0],0.0)),1.0))
		reward[1,0] = np.min((np.max((reward[1,0],0.0)),1.0))

		# # discount purseur if purseur on the boundary 
		# if np.any(s_1 == self.state_lims[0:6,:]):
		# 	reward[0,0] = 0.8 * reward[0,0]
		
		# # discount evader if evader on the boundary 
		# if np.any(s_1 == self.state_lims[6:,:]):
		# 	reward[1,0] = 0.8 * reward[1,0]

		# if False:
		# 	# discount purseur if evader is outside of heading cone
		# 	# from: https://www.mathworks.com/matlabcentral/answers/408012-how-to-check-if-a-3d-point-is-inside-a-3d-cone
		# 	heading_angle = 35 * np.pi / 180 # rad
		# 	u = s_1[3:6,:] / np.linalg.norm(s_1[3:6,0]) 
		# 	v = s_1[0:3,:]  
		# 	r = s_2[0:3,:]  
		# 	uvr = (r - v) / np.linalg.norm(r[:,0]-v[:,0])
		# 	angle = np.arccos(np.dot(uvr[:,0], u[:,0]))
		# 	if angle > heading_angle:
		# 		reward[0,0] = 0.8 * reward[0,0]

		if not self.in_elevation_cone(s):
			reward[0,0] = 0.8 * reward[0,0]

		return reward

	def normalized_reward(self,s,a): 
		reward = self.reward(s,a)
		return reward

	def step(self,s,a,dt):
		s_tp1 = np.zeros(s.shape)
		for robot in range(self.num_robots):
			Fd = np.eye(len(self.state_idxs[robot])) +  dt * self.Fc 
			Bd = dt * self.Bc 
			s_tp1[self.state_idxs[robot],:] = np.dot(Fd,s[self.state_idxs[robot],:]) + np.dot(Bd,a[self.action_idxs[robot],:])
		# apply state constraint bounds 
		for i in range(12):
			s_tp1[i,0] = np.min((np.max((s_tp1[i,0],self.state_lims[i,0])),self.state_lims[i,1]))
		# apply min speed bound
		s_tp1 = self.apply_min_speed(s_tp1)
		return s_tp1 

	def render(self,states=None,fig=None,ax=None):
		# states, np array in [nt x state_dim]
		
		if fig == None or ax == None:
			fig,ax = plotter.make_3d_fig()

		colors = ["orange", "purple"]

		lims = self.state_lims

		if states is not None:
			nt = states.shape[0]
			p1 = states[:,0:3,0] # nt x 3
			v1 = states[:,3:6,0] # nt x 3
			p2 = states[:,6:,0]  # nt x 3

			# labels
			ax.plot(np.nan, np.nan, marker="o", color=colors[0], alpha=0.5, label="Pursuer")
			ax.plot(np.nan, np.nan, marker="o", color=colors[1], alpha=0.5, label="Evader")

			# traj
			ax.plot(p1[:,0],p1[:,1],p1[:,2],
				color=colors[0], marker="o", markersize=0.5, alpha=0.25, linewidth=1)
			ax.plot(p2[:,0],p2[:,1],p2[:,2],
				color=colors[1], marker="o", markersize=0.5, alpha=0.25, linewidth=1)

			# line segments
			from mpl_toolkits.mplot3d.art3d import Line3DCollection
			segments = []
			for k in range(states.shape[0]):
				line = [(p1[k,0],p1[k,1],p1[k,2]), (p2[k,0],p2[k,1],p2[k,2])]
				segments.append(line)
			ln_coll = Line3DCollection(segments, 
				linewidth=0.2, colors='k', alpha=0.2)
			ax.add_collection(ln_coll)

			# projections
			ax.plot(lims[0,0]*np.ones(nt), p1[:,1], p1[:,2],\
				color=colors[0], alpha=0.5, linewidth=0.2, linestyle="--")
			ax.plot(p1[:,0], lims[1,1]*np.ones(nt), p1[:,2], \
				color=colors[0], alpha=0.5, linewidth=0.2, linestyle="--")
			ax.plot(p1[:,0], p1[:,1], lims[2,0]*np.ones(nt), \
				color=colors[0], alpha=0.5, linewidth=0.2, linestyle="--")

			ax.plot(lims[0,0]*np.ones(nt), p2[:,1], p2[:,2],\
				color=colors[1], alpha=0.5, linewidth=0.2, linestyle="--")
			ax.plot(p2[:,0], lims[1,1]*np.ones(nt), p2[:,2], \
				color=colors[1], alpha=0.5, linewidth=0.2, linestyle="--")
			ax.plot(p2[:,0], p2[:,1], lims[2,0]*np.ones(nt), \
				color=colors[1], alpha=0.5, linewidth=0.2, linestyle="--")

			if False:
				# heading cone 
				cone_length = 100 
				cone_angle = 35 * np.pi / 180
				if int(nt/5) > 1:
					cone_time_idxs = list(np.arange(0, nt, int(nt/5)))
					cone_time_idxs.append(-1)
				else:
					cone_time_idxs = [i for i in range(nt)]
				for idx in cone_time_idxs:
					A0 = p1[idx]
					R0 = 1.0
					A1 = p1[idx] + v1[idx] / np.linalg.norm(v1[idx]) * cone_length
					R1 = np.tan(cone_angle) * cone_length
					fig, ax = plotter.truncated_cone(A0, A1, R0, R1, fig, ax, color="gray", alpha=0.5)
				ax.plot(np.nan, np.nan, marker="o", color="gray", alpha=0.5, label="Heading Cone")

			# init space and full space
			fig, ax = plotter.plot_cube(self.init_lims[0:3,:], fig, ax, "red", 0.1)
			fig, ax = plotter.plot_cube(self.state_lims[0:3,:], fig, ax, "green", 0.1)
			ax.plot(np.nan, np.nan, marker="o", color="red", alpha=0.1, label=r"$X_0$")
			ax.plot(np.nan, np.nan, marker="o", color="green", alpha=0.1, label=r"$X$")

			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))
			ax.set_zlim((lims[1,0],lims[1,1]))
			ax.set_box_aspect((lims[0,1]-lims[0,0], lims[1,1]-lims[1,0], lims[1,1]-lims[1,0]))  
			# ax.set_zlim((lims[2,0],lims[2,1]))
			# ax.set_box_aspect((lims[0,1]-lims[0,0], lims[1,1]-lims[1,0], lims[2,1]-lims[2,0]))  

			ax.set_xlabel("x")
			ax.set_ylabel("y")
			ax.set_zlabel("z")

			ax.legend(loc='best')

		# make a second fig:
		if states is not None:
			fig2, ax2 = plotter.make_fig()
			p1 = states[:,0:3,0] # nt x 3
			p2 = states[:,6:9,0]  # nt x 3
			ds = np.linalg.norm(p1-p2, axis=1)

			bs = [self.in_elevation_cone(s) for s in states]

			ax2.plot(ds, color="blue")
			ax2.set_xlabel("time")
			ax2.set_ylabel("distance")

			ax3 = ax2.twinx()
			ax3.plot(bs, color="orange")
			ax3.set_ylabel("in elevation cone")

		return fig,ax 

	def is_terminal(self,state):
		return not self.is_valid(state)

	def is_valid(self,state):
		return contains(state,self.state_lims)

	def policy_encoding(self,state,robot):
		return state[0:6,:] - state[6:,:] 

	def value_encoding(self,state):
		return state[0:6,:] - state[6:,:]

	def plot_value_dataset(self, dataset, title):
		pass

	def plot_policy_dataset(self, dataset, title, robot):
		pass