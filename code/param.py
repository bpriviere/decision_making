
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example2" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT" # e.g. Empty, DARE, MCTS, PUCT, C_PUCT, ...

		# solver settings 
		if self.solver_name in ["PUCT","C_PUCT","PUCT_V2"]:
			self.number_simulations = 1000
			self.policy_oracle = None
			self.value_oracle = None
			self.search_depth = 10
			self.C_pw = 2.0
			self.alpha_pw = 0.5
			self.C_exp = 1.0
			self.alpha_exp = 0.25
			self.beta_policy = 0.0
			self.beta_value = 0.0
			self.vis_on = False

		# problem settings 
		if self.problem_name == "example1":
			self.t0 = 0
			self.tf = 20
			self.dt = 0.1
			pos_lim, vel_lim = 5,1 
			self.state_lims = np.array([
				[-pos_lim,pos_lim],
				[-pos_lim,pos_lim]
			])
			self.action_lims = np.array([
				[-vel_lim,vel_lim],
				[-vel_lim,vel_lim]
			])
			self.init_lims = np.copy(self.state_lims)

		elif self.problem_name == "example2":
			self.t0 = 0
			self.tf = 20
			self.dt = 0.1
			self.mass = 1
			pos_lim,vel_lim,acc_lim = 5,1,1
			self.state_lims = np.array([
				[-pos_lim,pos_lim],
				[-pos_lim,pos_lim],
				[-vel_lim,vel_lim],
				[-vel_lim,vel_lim]
			])
			self.action_lims = np.array([
				[-acc_lim,acc_lim],
				[-acc_lim,acc_lim]
			])
			self.init_lims = np.copy(self.state_lims)

		elif self.problem_name == "example3":
			self.t0 = 0
			self.tf = 20
			self.dt = 0.1
			self.desired_distance = 0.2
			self.state_control_weight = 0.0001
			self.g = 1.0 
			pos_lim = 10.0 
			vel_lim = 2.0 
			acc_lim = 1.0 / 2.0 
			rad_lim = 2 * np.pi 
			omega_lim = 2*np.pi / 100
			state_dim_per_robot = 7
			action_dim_per_robot = 3 
			num_robots = 2 
			self.state_lims = np.zeros((num_robots*state_dim_per_robot,2))
			self.init_lims = np.zeros((num_robots*state_dim_per_robot,2))
			self.action_lims = np.zeros((num_robots*action_dim_per_robot,2))
			for i in range(num_robots):
            	# s = [x,y,z,psi,gamma,phi,v], 
            	# a = [gammadot, phidot,vdot]
				state_shift = state_dim_per_robot * i 
				action_shift = action_dim_per_robot * i 
				self.state_lims[state_shift + np.arange(0,3),0] = -pos_lim
				self.state_lims[state_shift + np.arange(0,3),1] =  pos_lim
				self.state_lims[state_shift + 3,0] = -rad_lim
				self.state_lims[state_shift + 3,1] =  rad_lim
				self.state_lims[state_shift + 4,0] = -rad_lim
				self.state_lims[state_shift + 4,1] =  rad_lim
				self.state_lims[state_shift + 5,0] = -rad_lim
				self.state_lims[state_shift + 5,1] =  rad_lim
				self.state_lims[state_shift + 6,0] =  0.5*vel_lim
				self.state_lims[state_shift + 6,1] =  vel_lim

				self.action_lims[action_shift + np.arange(0,2),0] = -omega_lim
				self.action_lims[action_shift + np.arange(0,2),1] =  omega_lim
				self.action_lims[action_shift+2,0] = -acc_lim
				self.action_lims[action_shift+2,1] =  acc_lim
			
			self.init_lims = np.copy(self.state_lims)

	def to_dict(self):
		return self.__dict__