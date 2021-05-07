
import numpy as np 

class Param: 

	def __init__(self):

		# names 
		self.problem_name = "example7" # e.g. example1, example2, example3, ...
		self.solver_name = "PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "deterministic" # ["deterministic","gaussian"]

		# oracles 
		self.policy_oracle_paths = [None] # "/home/ben/projects/decision_making/current/models/model_policy_l19_i0.pt"
		# self.policy_oracle_paths = ["/home/ben/projects/decision_making/current/models/model_policy_l19_i0.pt"] 
		self.value_oracle_path = None # "/home/ben/projects/decision_making/current/models/model_value_l19.pt"
		# self.value_oracle_path = "/home/ben/projects/decision_making/current/models/model_value_l19.pt"
		
		# settings
		self.movie_on = False

		# solver settings 
		self.number_simulations = 100
		self.search_depth = 10
		self.C_pw = 2.0
		self.alpha_pw = 0.5
		self.C_exp = 1.0
		self.alpha_exp = 0.25
		self.beta_policy = 0.8
		self.beta_value = 1.0
		self.vis_on = True

	def to_dict(self):
		return self.__dict__