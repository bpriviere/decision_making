
import numpy as np 

class Param: 

	def __init__(self):

		# 
		self.parallel_on = True
		self.num_trials = 5 

		# names 
		self.problem_name = "example6" # e.g. example1, example2, example3, ...
		self.solver_name = "C_PUCT_V1" # e.g. Empty, DARE, PUCT_V0, C_PUCT_V0, PUCT_V1, ...
		self.value_oracle_name = "deterministic" # ["deterministic","gaussian"]
		self.policy_oracle_name = "gaussian" # ["deterministic","gaussian"]

		# oracles 
		self.policy_oracle_paths = [None]  
		# self.policy_oracle_paths = ["/home/ben/projects/decision_making/current/models/model_policy_l7_i{}.pt".format(i) for i in range(1)]
		# self.policy_oracle_paths = ["/home/ben/projects/decision_making/current/models/model_policy_l19_i0.pt"] 
		# self.policy_oracle_paths = ["/home/ben/projects/decision_making/saved/example6/models/model_policy_l19_i{}.pt".format(i) for i in range(1)] 
		# self.policy_oracle_paths = ["/home/ben/projects/decision_making/saved/example7/models/model_policy_l8_i{}.pt".format(i) for i in range(2)] 
		
		self.value_oracle_path = None 
		# self.value_oracle_path = "/home/ben/projects/decision_making/current/models/model_value_l7.pt"
		# self.value_oracle_path = "/home/ben/projects/decision_making/current/models/model_value_l19.pt"
		# self.value_oracle_path = "/home/ben/projects/decision_making/saved/example6/models/model_value_l19.pt"
		# self.value_oracle_path = "/home/ben/projects/decision_making/saved/example7/models/model_value_l8.pt"


		# settings
		self.movie_on = False
		self.pretty_plot_on = True

		# solver settings 
		self.number_simulations = 100
		self.search_depth = 50
		self.C_pw = 2.0
		self.alpha_pw = 0.5
		self.C_exp = 1.0
		self.alpha_exp = 0.25
		self.beta_policy = 0.75
		self.beta_value = 0.75
		self.vis_on = True

	def to_dict(self):
		return self.__dict__