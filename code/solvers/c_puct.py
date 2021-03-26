


# standard
import numpy as np 

# custom
import plotter 
from solvers.solver import Solver 
from cpp.build.bindings import cpp_search, Settings

class C_PUCT(Solver):

	def __init__(self,
		policy_oracle=None,\
		value_oracle=None,\
		search_depth=10,\
		number_simulations=1000,
		C_pw=2.0,
		alpha_pw=0.5,
		C_exp=1.0,
		alpha_exp=0.25,
		beta_policy=0.,
		beta_value=0.,
		vis_on=False,
		):
		super(C_PUCT, self).__init__()

		self.policy_oracle = policy_oracle 
		self.value_oracle = value_oracle 
		self.search_depth = search_depth 
		self.number_simulations = number_simulations 
		self.C_pw = C_pw 
		self.alpha_pw = alpha_pw 
		self.C_exp = C_exp 
		self.alpha_exp = alpha_exp 
		self.beta_policy = beta_policy 
		self.beta_value = beta_value
		self.vis_on = vis_on 

	def policy(self,problem,root_state):
		result = self.wrap_search(problem,root_state)
		py_action = np.zeros((problem.action_dim,1))
		py_action[:,0] = result.best_action
		return py_action

	def wrap_search(self,problem,root_state):

		if not problem.name == "example1":
			print('problem {} not supported'.format(problem.name))
			exit()

		settings = Settings()
		# settings.policy_oracle = self.policy_oracle # todo 
		# settings.value_oracle = self.value_oracle 
		settings.search_depth = self.search_depth 
		settings.num_nodes = self.number_simulations 
		settings.C_pw = self.C_pw 
		settings.alpha_pw = self.alpha_pw 
		settings.C_exp = self.C_exp 
		settings.alpha_exp = self.alpha_exp 
		settings.beta_policy = self.beta_policy 
		settings.beta_value = self.beta_value
		settings.vis_on = self.vis_on 

		# temp 
		import random 
		seed = random.randint(1,1000)

		result = cpp_search(root_state,settings,seed)

		if settings.vis_on: 
			tree_state = result.tree 
			plotter.plot_tree_state(problem,tree_state,zoom_on=False)

		return result

# # standard
# import numpy as np 

# # custom
# import plotter 
# from solvers.solver import Solver 
# from cpp.build.bindings import cpp_search, Settings, PUCT_Wrapper

# class C_PUCT(Solver):

# 	def __init__(self,
# 		policy_oracle=None,\
# 		value_oracle=None,\
# 		search_depth=10,\
# 		number_simulations=1000,
# 		C_pw=2.0,
# 		alpha_pw=0.5,
# 		C_exp=1.0,
# 		alpha_exp=0.25,
# 		beta_policy=0.,
# 		beta_value=0.,
# 		vis_on=False,
# 		):
# 		super(C_PUCT, self).__init__()

# 		settings = Settings()
# 		settings.search_depth = search_depth 
# 		settings.num_nodes = number_simulations 
# 		settings.C_pw = C_pw 
# 		settings.alpha_pw = alpha_pw 
# 		settings.C_exp = C_exp 
# 		settings.alpha_exp = alpha_exp 
# 		settings.beta_policy = beta_policy 
# 		settings.beta_value = beta_value
# 		settings.vis_on = vis_on 

# 		self.puct_wrapper = PUCT_Wrapper(settings)

# 	def policy(self,problem,root_state):
# 		result = self.wrap_search(problem,root_state)
# 		py_action = np.zeros((problem.action_dim,1))
# 		py_action[:,0] = result.best_action
# 		return py_action

# 	def wrap_search(self,problem,root_state):

# 		if not problem.name == "example1":
# 			print('problem {} not supported'.format(problem.name))
# 			exit()

# 		result = self.puct_wrapper(root_state)

# 		if settings.vis_on: 
# 			tree_state = result.tree 
# 			plotter.plot_tree_state(problem,tree_state,zoom_on=True)

# 		return result