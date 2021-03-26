
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example1" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT" # e.g. Empty, DARE, MCTS, PUCT, C_PUCT, ...

		# solver settings 
		if self.solver_name in ["PUCT","C_PUCT","PUCT_V2"]:
			self.number_simulations = 50
			self.policy_oracle = None
			self.value_oracle = None
			self.search_depth = 10
			self.C_pw = 2.0
			self.alpha_pw = 0.5
			self.C_exp = 1.0
			self.alpha_exp = 0.25
			self.beta_policy = 0.0
			self.beta_value = 0.0
			self.vis_on = True

		# problem settings 
		if self.problem_name == "example1":
			self.t0 = 0
			self.tf = 10
			self.dt = 0.1
			self.pos_lim = 5
			self.vel_lim = 1
		elif self.problem_name == "example2":
			self.t0 = 0
			self.tf = 10
			self.dt = 0.1
			self.pos_lim = 5
			self.vel_lim = 1
			self.acc_lim = 1
			self.mass = 1		
		elif self.problem_name == "example3":
			self.t0 = 0
			self.tf = 20
			self.dt = 0.1
			self.pos_lim = 2
			self.vel_lim = 2
			self.acc_lim = 1.0
			self.rad_lim = 2*np.pi
			self.omega_lim = 2*np.pi / 10
			self.desired_distance = 0.2
			self.state_control_weight = 0.1
			self.g = 3.0


	def to_dict(self):
		return self.__dict__