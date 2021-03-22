


# standard
import numpy as np 

# custom
from solvers.solver import Solver 
from cpp.bindings import cpp_search

class C_PUCT(Solver):

	def __init__(self):
		pass 

	def policy(self,problem,root_state):
		root_node = self.search_wrapper(problem,root_state)
		most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
		best_action = root_node.edges[most_visited_child]
		return best_action 

	def search_wrapper(self,problem,root_state):
		cpp_root_node = cpp_search(problem,root_state)
		root_node = Node(root_state,None,problem.num_robots)

		return root_node 





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

