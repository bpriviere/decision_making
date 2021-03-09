
# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 

# not working 
# from https://arxiv.org/abs/1902.05213
class MCTS(Solver):

	def __init__(self):
		super(MCTS, self).__init__()
		self.search_depth = 10
		self.number_simulations = 100 

	def policy(self,problem,root_state):
		return self.search(problem,root_state)

	def calc_constants(self,search_depth):
		
		delta_min = 1

		etas = 1/2 * np.ones(search_depth) 
		alphas = np.zeros(search_depth)
		epsilons = np.zeros(search_depth)
		betas = np.zeros(search_depth)

		alphas[0] = 2 
		for h in range(search_depth-1):
			alphas[h+1] = alphas[h] / (etas[0]*(1-etas[0])) + 1 
			epsilons[h] = alphas[h+1] - 1 
			betas[h] = (delta_min / 2) ** epsilons[h]

		# ?? 
		epsilons[search_depth] = 1
		betas[search_depth] = 1 

		return etas, alphas, epsilons, betas

	def select_action(curr_node, etas, alphas, epsilons, betas):

		

		return action 

	def search(self,\
		problem,
		root_state,\
		value_oracle=None,\
		search_depth=None,\
		number_simulations=None,
		vis_on=False): 
		
		if search_depth is None: 
			search_depth = self.search_depth
		if number_simulations is None: 
			number_simulations = self.number_simulations

		# constants 
		etas, alphas, epsilons, betas = self.calc_constants(search_depth) 

		alphas = self.calculate_alphas()
		epsilons = self.calculate_epsilons()
		betas = self.calculate_betas() 

		# init tree 
		root_node = Node(root_state)

		# search 
		for t in range(1,number_simulations):

			curr_node = root_node
			search_path = [(curr_node,None,None)]

			# simulate 
			for h in range(search_depth):

				s_h = curr_node.state
				a_hp1 = self.select_action(curr_node, etas, alphas, epsilons, betas)
				r_hp1 = problem.calculate_reward(s_h,a_hp1)
				s_hp1 = problem.step(s_h,a_hp1)

				if a_hp1 in curr_node.children.keys():
					curr_node = curr_node.children[a_hp1] 
				else:
					parent_node = curr_node
					curr_node = Node(s_hp1)
					parent_node.add_child(curr_node,a_hp1)
				
				search_path.append((curr_node,a_hp1,r_hp1)) # s_hp1,a_hp1,r_hp1

			# query oracle at leaf node 
			if value_oracle is None: 
				v_H = 0
			curr_node.value = v_H 
			leaf_node = curr_node 

			# update statistics along search path 
			for h in range(search_depth):
				node_h = search_path[h][0]
				node_hp1 = search_path[h+1][0]
				a_hp1 = search_path[h+1][1]
				r_hp1 = search_path[h+1][2]

				node_hp1.num_visits = node_hp1.num_visits + 1 
				node_hp1.edge_values[a_hp1] = node_hp1.edge_values[a_hp1] + r_hp1 
				
				update_value = node_h.value 
				for hprime in range(h+1,search_depth):
					update_value = update_value + search_path[hprime][2] * problem.gamma ** (h_prime - h - 1)
				node_hp1.value = update_value + leaf_node.value * problem.gamma ** (search_depth - h) 

		# choose best action 
		child_value_actions = [(c.value,a) for (a,c) in root_node[children].items()]
		best_action = max(best_child)[1]

		return best_action 



class Node: 

	def __init__(self,state):
		self.state = state 
		self.num_visits = 0 
		self.value = 0 
		self.children = dict() 
		self.edge_values = dict() 

	def add_child(self,child_node,action):
		self.children[action] = child_node 
		self.edge_values[action] = 0 