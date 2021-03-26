

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

	def sample_action():
		exit("sample_action needs to be overwritten")

	def sample_state():
		exit("sample_action needs to be overwritten")	

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

	def initialize(self):
		exit("initialize needs to be overwritten")

	def policy_encoding(self,state,robot):
		exit("policy_encoding needs to be overwritten")		

	def value_encoding(self,state):
		exit("value_encoding needs to be overwritten")

	def render(self,states):
		exit("render needs to be overwritten")


