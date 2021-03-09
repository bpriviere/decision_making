
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 
import plotter


# Polynomial Upper Confidence Trees (PUCT):
# from https://hal.inria.fr/hal-00835352/document
class PUCT(Solver):

	def __init__(self):
		super(PUCT, self).__init__()
		self.search_depth = 10
		self.number_simulations = 1000
		self.C_pw = 2.0 
		self.alpha_pw = 0.5
		self.C_exp = 1.0
		self.alpha_exp = 0.25


	def policy(self,problem,root_state):
		return self.search(problem,root_state,vis_on=True)
		# return self.search(problem,root_state)


	def select_node(self,root_node,problem):
		curr_node = root_node 
		while not problem.is_terminal(curr_node.state):
			if self.is_expanded(curr_node):
				curr_node = self.best_child(curr_node)
			else: 
				return curr_node 
		return curr_node


	def expand_node(self,parent_node,problem):
		action = problem.A.sample()
		next_state = problem.step(parent_node.state,action)
		child_node = Node(next_state,parent_node)
		parent_node.add_child(child_node,action)
		return child_node


	def is_expanded(self,curr_node):
		max_children = np.ceil(self.C_pw * (curr_node.num_visits ** self.alpha_pw));
		return len(curr_node.children) > max_children 


	def best_child(self,curr_node):
		best_c = None
		best_value = 0
		for child in curr_node.children: 
			if child.num_visits == 0: 
				return child
			value = child.total_value / child.num_visits + \
				self.C_exp * np.sqrt((child.num_visits ** self.alpha_exp)/curr_node.num_visits)
			if value > best_value: 
				best_value = value 
				best_c = child 
		return best_c 


	def default_policy(self,node,problem,search_depth):
		value = 0 
		depth = node.calc_depth()
		curr_state = node.state 
		while not problem.is_terminal(curr_state) and depth < search_depth:
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


	def search(self,\
		problem,
		root_state,\
		policy_oracle=None,\
		value_oracle=None,\
		search_depth=None,\
		number_simulations=None,
		vis_on=False): 
		
		if search_depth is None: 
			search_depth = self.search_depth
		if number_simulations is None: 
			number_simulations = self.number_simulations

		# check validity
		if problem.is_terminal(root_state):
			print('root node is terminal')
			return None

		# init tree 
		root_node = Node(root_state,None)

		# search 
		for t in range(number_simulations):
			parent_node = self.select_node(root_node,problem)
			child_node = self.expand_node(parent_node,problem)
			value = self.default_policy(child_node,problem,search_depth)
			self.backup(child_node,value)

		if vis_on: 
			tree_state = self.export_tree(root_node)
			plotter.plot_tree_state(problem,tree_state)

		# return most visited root action 
		most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
		best_action = root_node.edges[most_visited_child]

		# return most valued child
		# most_valued_child = root_node.children[np.argmax([c.total_value for c in root_node.children])]
		# best_action = root_node.edges[most_valued_child]

		return best_action 


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

	def __init__(self,state,parent):
		self.state = state 
		self.parent = parent 
		self.num_visits = 0 
		self.total_value = 0 
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
