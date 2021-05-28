


# standard 
import numpy as np 

# custom 
from solvers.solver import Solver 
from solvers.policy_solver import PolicySolver
import plotter


# Polynomial Upper Confidence Trees (PUCT) pseudocode from PolyHOOT
class PUCT_V1(Solver):

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
		super(PUCT_V1, self).__init__()

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
		self.solver_name = "PUCT_V1"


	def policy(self,problem,root_state):
		action = np.zeros((problem.action_dim,1))
		for robot in range(problem.num_robots):
			robot_action_idxs = problem.action_idxs[robot] 
			root_node = self.search(problem,root_state,turn=robot)
			if root_node.success:
				most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
				action[robot_action_idxs,0] = root_node.edges[most_visited_child][robot_action_idxs,0]
		return action

	def expand_node(self,parent_node,problem):

		if not all(x is None for x in self.policy_oracle) and np.random.uniform() < self.beta_policy:
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
			value = self.value_oracle.eval(problem,node.state) 
		else: 
			value = np.zeros((problem.num_robots,1))
			if True:
				depth = 0 
				curr_state = node.state 
				while not problem.is_terminal(curr_state) and depth < self.search_depth:
					action = problem.sample_action()
					next_state = problem.step(curr_state,action,problem.dt)
					value += (problem.gamma ** depth) * problem.normalized_reward(curr_state,action)
					curr_state = next_state 
					depth += 1
		return value 


	def calc_value(self,rewards,start_depth,total_depth,gamma,num_robots):
		value = np.zeros((num_robots,1)) 
		for d in range(start_depth,total_depth):
			value += rewards[d] * gamma ** d
		return value 


	def search(self,problem,root_state,turn=0): 
		
		# init tree 
		root_node = Node(root_state,None,problem.num_robots)
		root_node.success = False

		# check validity
		if problem.is_terminal(root_state):
			return root_node

		if not all(x is None for x in self.policy_oracle):
			self.policy_solver = PolicySolver(self.policy_oracle)

		# search 
		for t in range(self.number_simulations):
			
			curr_node = root_node
			rewards = [] 
			path = [] 

			# collect data
			for d in range(self.search_depth):
				robot = (d+turn) % problem.num_robots  
				if self.is_expanded(curr_node):
					child_node = self.best_child(curr_node,robot) 
				else:
					child_node = self.expand_node(curr_node,problem)

				path.append(curr_node)
				rewards.append(problem.normalized_reward(curr_node.state,curr_node.edges[child_node]))
				curr_node = child_node 

				if problem.is_terminal(child_node.state):
					break 

			rewards.append(self.default_policy(child_node,problem))
			path.append(curr_node)

			# backpropagate 
			for d,node in enumerate(path):
				node.total_value += self.calc_value(rewards,d,len(path),problem.gamma,problem.num_robots)
				node.num_visits += 1 

		root_node.success = True
		root_node.value = root_node.total_value / root_node.num_visits

		if self.vis_on: 
			tree_state = self.export_tree(root_node)
			plotter.plot_tree_state(problem,tree_state,zoom_on=False)

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
