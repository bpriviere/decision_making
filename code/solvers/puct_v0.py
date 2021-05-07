
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 
from solvers.policy_solver import PolicySolver, ValueSolver
import plotter


# Polynomial Upper Confidence Trees (PUCT) psuedocode from Browne 2012 MCTS Survey
class PUCT_V0(Solver):

	def __init__(self,
		policy_oracle=None,\
		value_oracle=None,\
		search_depth=10,\
		number_simulations=1000,
		C_pw = 2.0,
		alpha_pw = 0.5,
		C_exp = 1.0,
		alpha_exp = 0.25,
		beta_policy = 0.,
		beta_value = 0.,
		vis_on=False,
		):
		super(PUCT_V0, self).__init__()

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
		self.solver_name = "PUCT_V0"


	def policy(self,problem,root_state):
		action = np.zeros((problem.action_dim,1))
		for robot in range(problem.num_robots): 
			action_idxs = robot * problem.action_dim_per_robot + \
				np.arange(problem.action_dim_per_robot)
			root_node = self.search(problem,root_state,turn=robot)
			most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
			action[action_idxs,0] = root_node.edges[most_visited_child][action_idxs,0]
		return action


	def select_node(self,root_node,problem,turn):
		curr_node = root_node 
		depth = 0 
		while not problem.is_terminal(curr_node.state):
			robot = (depth+turn) % problem.num_robots
			if self.is_expanded(curr_node):
				curr_node = self.best_child(curr_node,robot)
				depth += 1 
			else: 
				return curr_node 
		return curr_node


	def expand_node(self,parent_node,problem):
		if self.policy_oracle is not None and np.random.uniform() < self.beta_policy:
			action = self.policy_solver.policy(problem,parent_node.state)
		else: 
			action = problem.sample_action()
		next_state = problem.step(parent_node.state,action,problem.dt)
		child_node = Node(next_state,parent_node,problem.num_robots)
		parent_node.add_child(child_node,action)
		return child_node


	def is_expanded(self,curr_node):
		max_children = np.ceil(self.C_pw * (curr_node.num_visits ** self.alpha_pw))
		return len(curr_node.children) > max_children 


	def best_child(self,curr_node,robot):
		best_c = None
		best_value = -np.inf 
		for child in curr_node.children: 
			if child.num_visits == 0: 
				return child
			value = child.total_value[robot] / child.num_visits + \
				self.C_exp * np.sqrt((curr_node.num_visits ** self.alpha_exp)/child.num_visits)
			if value > best_value: 
				best_value = value 
				best_c = child 
		return best_c 


	def default_policy(self,node,problem):
		if self.value_oracle is not None and np.random.uniform() < self.beta_value:
			value = self.value_solver(problem,node.state) 
		else: 
			value = 0 
			depth = node.calc_depth()
			curr_state = node.state 
			while not problem.is_terminal(curr_state) and depth < self.search_depth:
				action = problem.sample_action()
				next_state = problem.step(curr_state,action,problem.dt)
				value += (problem.gamma ** depth) * problem.normalized_reward(curr_state,action)
				curr_state = next_state 
				depth += 1
		return value 


	def backup(self,node,value):
		curr_node = node 
		while curr_node.parent is not None: 
			curr_node.num_visits += 1 
			curr_node.total_value += value 
			curr_node = curr_node.parent 
		curr_node.num_visits += 1 # update root 
		curr_node.total_value += value 
		return 


	def search(self,problem,root_state,turn=0): 
		
		# check validity
		if problem.is_terminal(root_state):
			# print('root node is terminal')
			return None

		# init tree 
		root_node = Node(root_state,None,problem.num_robots)
		root_node.success = False

		# init heuristics
		if self.policy_oracle is not None: 
			self.policy_solver = PolicySolver(problem,self.policy_oracle)
		if self.value_oracle is not None: 
			self.value_solver = ValueSolver(problem,self.value_oracle)

		# search 
		for t in range(self.number_simulations):
			parent_node = self.select_node(root_node,problem,turn)
			child_node = self.expand_node(parent_node,problem)
			value = self.default_policy(child_node,problem)
			self.backup(child_node,value)

		if self.vis_on: 
			tree_state = self.export_tree(root_node)
			plotter.plot_tree_state(problem,tree_state,zoom_on=True)

		root_node.success = True
		root_node.value = root_node.total_value / root_node.num_visits
		return root_node


	def export_tree(self,root_node):
		# returns np array in [num_nodes x state_dim + 1]
		tree = []
		to_add = [(root_node,-1)]
		while len(to_add) > 0:
			curr_node,parent_idx = to_add.pop(0)
			tree.append(np.append(curr_node.state,parent_idx))
			parent_idx = len(tree) - 1
			for child in curr_node.children: 
				to_add.append((child,parent_idx))
		tree = np.array(tree)
		return tree 
		

	def get_child_distribution(self,root_node):
		actions,num_visits = list(zip(*[(a.squeeze(),c.num_visits) for (c,a) in root_node.edges.items()]))
		return actions,num_visits


class Node: 

	def __init__(self,state,parent,num_robots):
		self.state = state 
		self.parent = parent 
		self.num_visits = 0 
		self.total_value = np.zeros((num_robots,1))
		self.children = []
		self.edges = dict()


	def add_child(self,child_node,action):
		self.children.append(child_node)
		self.edges[child_node] = action 


	def calc_depth(self):
		curr_node = self 
		depth = 0 
		while curr_node.parent is not None: 
			curr_node = curr_node.parent 
			depth += 1 
		return depth 
