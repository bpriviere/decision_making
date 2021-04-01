

# standard
import numpy as np 

# custom
import plotter 
from solvers.solver import Solver 
from cpp.build.bindings import search, PUCT_Wrapper, Result, Problem_Wrapper

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
		result = self.wrap_search(problem,root_state)
		py_action = np.zeros((problem.action_dim,1))
		py_action[:,0] = result.best_action
		return py_action

	def wrap_search(self,problem,root_state):

		if not problem.name == "example1":
			print('problem {} not supported'.format(problem.name))
			exit()
		else: 
			cpp_problem = Problem_Wrapper()

		result = search(self.w_puct,cpp_problem,root_state)

		if self.vis_on: 
			tree_state = result.tree 
			plotter.plot_tree_state(problem,tree_state,zoom_on=True)

		return result