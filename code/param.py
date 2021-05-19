
import numpy as np 

class Param: 

	def __init__(self):

		# 
		self.parallel_on = True
		self.num_trials = 5 

		# names 
		self.problem_name = "example9" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "gaussian" # ["deterministic","gaussian"]

		# oracles 
		oracles_on = False
		dirname = "../current"
		n = 2 # num robots 
		l = 0 # learning iteration 
		if oracles_on:
			self.policy_oracle_paths = ["{}/model_policy_l{}_i{}.pt".format(dirname,l,i) for i in range(n)]	
			self.value_oracle_path = "{}/models/model_value_l{}.pt".format(dirname,l)
		else:	
			self.policy_oracle_paths = [None]  
			self.value_oracle_path = None 

		# settings
		self.movie_on = False
		self.pretty_plot_on = True

		# solver settings 
		self.number_simulations = 1000
		self.search_depth = 50
		self.C_pw = 2.0
		self.alpha_pw = 0.5
		self.C_exp = 1.0
		self.alpha_exp = 0.25
		self.beta_policy = 0.75
		self.beta_value = 0.75
		self.vis_on = False

	def to_dict(self):
		return self.__dict__