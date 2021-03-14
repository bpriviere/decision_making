
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 
from solvers.policy_solver import PolicySolver, ValueSolver
import plotter


# Polynomial Upper Confidence Trees (PUCT):
# from https://hal.inria.fr/hal-00835352/document
class PUCT(Solver):

	def __init__(self,
		policy_oracle=None,\
		value_oracle=None,\
		search_depth=10,\
		number_simulations=10000,
		C_pw = 2.0,
		alpha_pw = 0.5,
		C_exp = 1.0,
		alpha_exp = 0.25,
		beta_policy = 0.5,
		beta_value = 0.5,
		vis_on=False,
		):
		super(PUCT, self).__init__()

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
		root_node = self.search(problem,root_state)
		most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
		best_action = root_node.edges[most_visited_child]
		return best_action 


	def select_node(self,root_node,problem,robot):
		curr_node = root_node 
		while not problem.is_terminal(curr_node.state):
			if self.is_expanded(curr_node):
				curr_node = self.best_child(curr_node,robot)
			else: 
				return curr_node 
		return curr_node


	def expand_node(self,parent_node,problem):
		if self.policy_oracle is not None and np.random.uniform() < self.beta_policy:
			action = self.policy_solver.policy(problem,parent_node.state)
		else: 
			action = problem.A.sample()
		next_state = problem.step(parent_node.state,action)
		child_node = Node(next_state,parent_node,problem.num_robots)
		parent_node.add_child(child_node,action)
		return child_node


	def is_expanded(self,curr_node):
		max_children = np.ceil(self.C_pw * (curr_node.num_visits ** self.alpha_pw));
		return len(curr_node.children) > max_children 


	def best_child(self,curr_node,robot):
		best_c = None
		best_value = -np.inf 
		for child in curr_node.children: 
			if child.num_visits == 0: 
				return child
			value = child.total_value[robot] / child.num_visits + \
				self.C_exp * np.sqrt((child.num_visits ** self.alpha_exp)/curr_node.num_visits)
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
				action = problem.A.sample()
				next_state = problem.step(curr_state,action)
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


	def search(self,problem,root_state): 
		
		# check validity
		if problem.is_terminal(root_state):
			# print('root node is terminal')
			return None

		# init tree 
		root_node = Node(root_state,None,problem.num_robots)

		# init heuristics
		if self.policy_oracle is not None: 
			self.policy_solver = PolicySolver(problem,self.policy_oracle)
		if self.value_oracle is not None: 
			self.value_solver = ValueSolver(problem,self.value_oracle)

		# search 
		for t in range(self.number_simulations):
			robot = t % problem.num_robots
			parent_node = self.select_node(root_node,problem,robot)
			child_node = self.expand_node(parent_node,problem)
			value = self.default_policy(child_node,problem)
			self.backup(child_node,value)

		if self.vis_on: 
			tree_state = self.export_tree(root_node)
			plotter.plot_tree_state(problem,tree_state)

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
		


class Node: 

	def __init__(self,state,parent,num_robots):
		self.state = state 
		self.parent = parent 
		self.num_visits = 0 
		self.total_value = np.zeros(num_robots) 
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
