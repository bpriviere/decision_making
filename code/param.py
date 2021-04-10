
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example2" # e.g. example1, example2, example3, ...
		self.solver_name = "PUCT" # e.g. Empty, DARE, MCTS, PUCT, C_PUCT, ...

		# solver settings 
		if self.solver_name in ["PUCT","C_PUCT","PUCT_V2"]:
			self.number_simulations = 100
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

	def to_dict(self):
		return self.__dict__