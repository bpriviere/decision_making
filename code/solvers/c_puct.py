

# standard
import numpy as np 
import torch

# custom
import plotter 
from solvers.solver import Solver 
from cpp.build.bindings import cpp_search, Solver_Result, Solver_Settings, Solver_Wrapper, Problem_Settings, Problem_Wrapper
from cpp.build.bindings import Policy_Network_Wrapper, Value_Network_Wrapper


class C_PUCT(Solver):

	def __init__(self,
		policy_oracle=[None],\
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
		solver_name=None,
		):
		super(C_PUCT, self).__init__()
		
		self.policy_oracle = self.create_cpp_policy_oracle(policy_oracle)
		self.value_oracle = self.create_cpp_value_oracle(value_oracle)

		self.vis_on = vis_on 

		self.solver_settings = Solver_Settings()
		self.solver_settings.number_simulations = number_simulations
		self.solver_settings.search_depth = search_depth
		self.solver_settings.C_pw = C_pw
		self.solver_settings.alpha_pw = alpha_pw
		self.solver_settings.C_exp = C_exp
		self.solver_settings.alpha_exp = alpha_exp
		self.solver_settings.beta_policy = beta_policy
		self.solver_settings.beta_value = beta_value
		self.solver_name = solver_name
		self.solver_wrapper = Solver_Wrapper(solver_name,self.solver_settings,self.policy_oracle,self.value_oracle)


	def policy(self,problem,root_state):
		py_action = np.zeros((problem.action_dim,1))
		for robot in range(problem.num_robots): 
			robot_action_idxs = problem.action_idxs[robot]
			result = self.search(problem,root_state,turn=robot)
			py_action[robot_action_idxs,0] = result.best_action[robot_action_idxs]

			# exit()

		if self.solver_name in ["C_PUCT_V2"]:
			py_action = np.append(py_action,np.array(result.best_action[-1],ndmin=2),axis=0)

		return py_action

	def search(self,problem,root_state,turn=0):

		# problem settings 
		problem_settings = Problem_Settings()
		problem_settings.state_dim = problem.state_dim
		problem_settings.action_dim = problem.action_dim
		problem_settings.num_robots = problem.num_robots
		problem_settings.timestep = problem.dt
		problem_settings.tf = problem.tf
		problem_settings.gamma = problem.gamma
		problem_settings.r_max = problem.r_max
		problem_settings.r_min = problem.r_min
		problem_settings.state_lims = problem.state_lims
		problem_settings.action_lims = problem.action_lims 
		problem_settings.state_idxs = problem.state_idxs
		problem_settings.action_idxs = problem.action_idxs
		problem_settings.init_lims = problem.init_lims 
		problem_settings.state_control_weight = problem.state_control_weight

		if problem.name == "example2":
			problem_settings.mass = problem.mass
		elif problem.name == "example3":
			problem_settings.g = problem.g 		
			problem_settings.desired_distance = problem.desired_distance
		elif problem.name == "example4":
			problem_settings.mass = problem.mass
			problem_settings.desired_distance = problem.desired_distance
		elif problem.name == "example5":
			problem_settings.m1 = problem.m1
			problem_settings.m2 = problem.m2
			problem_settings.c1 = problem.c1
			problem_settings.c2 = problem.c2
		elif problem.name in ["example6","example11"]:
			problem_settings.obstacles = problem.obstacles
			problem_settings.desired_distance = problem.desired_distance
			problem_settings.desired_state = problem.s_des
		elif problem.name == "example8":
			problem_settings.desired_distance = problem.desired_distance
		elif problem.name in ["example9","example12"]:
			problem_settings.desired_distance = problem.desired_distance
			problem_settings.R = problem.R
			problem_settings.c1 = problem.w1
			problem_settings.c2 = problem.w2

		if problem.name not in ["example{}".format(i) for i in [1,2,3,4,5,6,8,9,10,11,12]]:
			print("problem not supported")
			exit()

		# problem 
		problem_wrapper = Problem_Wrapper(problem.name,problem_settings)

		# 
		# print('search')
		result = cpp_search(problem_wrapper,self.solver_wrapper,root_state,turn)
		# print('done')

		if self.vis_on: 
			tree_state = result.tree 
			plotter.plot_tree_state(problem,tree_state,zoom_on=True)

		return result

	def get_child_distribution(self,result):
		mat = result.child_distribution;
		actions = mat[:,:-1].tolist()
		num_visits = mat[:,-1].tolist()
		return actions,num_visits


	def create_cpp_policy_oracle(self,policy_oracle):
		policy_wrappers = []
		for py_policy_oracle in policy_oracle:
			cpp_policy_wrapper = Policy_Network_Wrapper()
			if py_policy_oracle is not None:
				cpp_policy_wrapper.initialize(py_policy_oracle.name)
				parameter_dict = torch.load(py_policy_oracle.path)
				# assume only one feedforward neural network, named psi with weights
				self.loadFeedForwardNetworkWeights(cpp_policy_wrapper,parameter_dict,"psi")
			policy_wrappers.append(cpp_policy_wrapper)
		return policy_wrappers

	def create_cpp_value_oracle(self,value_oracle):
		cpp_value_wrapper = Value_Network_Wrapper()
		if value_oracle is not None:
			cpp_value_wrapper.initialize(value_oracle.name)
			parameter_dict = torch.load(value_oracle.path)
			# assume only one feedforward neural network, named psi with weights
			self.loadFeedForwardNetworkWeights(cpp_value_wrapper,parameter_dict,"psi")
		return cpp_value_wrapper


	def loadFeedForwardNetworkWeights(self, policy_wrapper, state_dict, name):
		l = 0
		while True:
			key1 = "{}.layers.{}.weight".format(name, l)
			key2 = "{}.layers.{}.bias".format(name, l)
			if key1 in state_dict and key2 in state_dict:
				policy_wrapper.addLayer(state_dict[key1].numpy(), state_dict[key2].numpy())
			else:
				break
			l += 1
