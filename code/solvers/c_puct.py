

# standard
import numpy as np 

# custom
import plotter 
from solvers.solver import Solver 
from cpp.build.bindings import cpp_search, PUCT_Wrapper, Result, Problem_Wrapper, Problem_Settings

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
		
		self.w_puct = PUCT_Wrapper(
			number_simulations,
			search_depth,
			C_pw,
			alpha_pw,
			C_exp,
			alpha_exp,
			beta_policy,
			beta_value,
			vis_on,
		)

	def policy(self,problem,root_state):
		result = self.search(problem,root_state)
		py_action = np.zeros((problem.action_dim,1))
		py_action[:,0] = result.best_action
		return py_action

	def search(self,problem,root_state):

		problem_settings = Problem_Settings()
		problem_settings.timestep = problem.dt
		problem_settings.gamma = problem.gamma
		problem_settings.r_max = problem.r_max
		problem_settings.state_lims = problem.state_lims
		problem_settings.action_lims = problem.action_lims 
		problem_settings.init_lims = problem.init_lims 

		if problem.name == "example1":
			problem_settings.state_control_weight = problem.state_control_weight
		elif problem.name == "example2":
			problem_settings.mass = problem.mass
			problem_settings.state_control_weight = problem.state_control_weight 
		elif problem.name == "example3":
			problem_settings.g = problem.g 		
			problem_settings.desired_distance = problem.desired_distance
			problem_settings.state_control_weight = problem.state_control_weight
		elif problem.name == "example4":
			problem_settings.mass = problem.mass
			problem_settings.desired_distance = problem.desired_distance
			problem_settings.state_control_weight = problem.state_control_weight
		else: 
			print("problem not supported")
			exit()

		cpp_problem = Problem_Wrapper(problem.name,problem_settings)

		result = cpp_search(self.w_puct,cpp_problem,root_state)

		if self.vis_on: 
			tree_state = result.tree 
			plotter.plot_tree_state(problem,tree_state,zoom_on=True)

		return result