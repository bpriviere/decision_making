
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example1" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...

		# settings
		self.movie_on = False

		# solver settings 
		if self.solver_name in ["PUCT_V0","PUCT_V1","C_PUCT_V0","C_PUCT_V1"]:
			self.number_simulations = 10
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