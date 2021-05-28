
from util import sample_vector

class Problem: 

	def __init__(self):
		self.num_robots = None
		self.gamma = None
		self.state_dim = None 
		self.state_lims = None 
		self.action_dim = None
		self.action_lims = None 
		self.position_idx = None 
		self.dt = None
		self.times = None  
		self.policy_encoding_dim = None
		self.value_encoding_dim = None
		self.name = None

	def sample_action(self):
		return sample_vector(self.action_lims)

	def sample_state(self):
		return sample_vector(self.state_lims)

	def initialize(self):
		valid = False
		while not valid:
			state = sample_vector(self.init_lims)
			valid = not self.is_terminal(state)
		return state

	def reward(self,state,action):
		exit("reward needs to be overwritten")

	def normalized_reward(self,state,action): 
		exit("normalized_reward needs to be overwritten")		

	def step(self,state,action):
		exit("step needs to be overwritten")

	def render(self,states):
		exit("render needs to be overwritten")

	def is_terminal(self,state):
		exit("is_terminal needs to be overwritten")

	def policy_encoding(self,state,robot):
		exit("policy_encoding needs to be overwritten")		

	def value_encoding(self,state):
		exit("value_encoding needs to be overwritten")

	def render(self,states):
		exit("render needs to be overwritten")


def get_problem(problem_name):

	# 2d single integrator regulator
	if problem_name == "example1":
		from problems.example1 import Example1
		problem = Example1()

	# 2d double integrator regulator
	elif problem_name == "example2":
		from problems.example2 import Example2
		problem = Example2()

	# 3d dubins uncooperative tracking 
	elif problem_name == "example3":
		from problems.example3 import Example3
		problem = Example3()

	# 3d double integrator uncooperative tracking 
	elif problem_name == "example4":
		from problems.example4 import Example4
		problem = Example4() 

	# game of atrition 
	elif problem_name == "example5":
		from problems.example5 import Example5
		problem = Example5()

	# bugtrap: 2d single integrator with obstacles 
	elif problem_name == "example6":
		from problems.example6 import Example6
		problem = Example6() 

	# 2d double integrator uncooperative tracking 
	elif problem_name == "example7":
		from problems.example7 import Example7
		problem = Example7() 

	# 2d single integrator pursuit evasion 
	elif problem_name == "example8":
		from problems.example8 import Example8
		problem = Example8() 

	# homicidal chauffeur problem
	elif problem_name == "example9":
		from problems.example9 import Example9
		problem = Example9() 

	# dummy game problem 
	elif problem_name == "example10":
		from problems.example10 import Example10
		problem = Example10() 

	# multiscale bugtrap: 2d single integrator with obstacles 
	elif problem_name == "example11":
		from problems.example11 import Example11
		problem = Example11() 

	# modified homicidal chauffer  
	elif problem_name == "example12":
		from problems.example12 import Example12
		problem = Example12() 

	return problem 