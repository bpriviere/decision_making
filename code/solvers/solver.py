

class Solver: 

	def __init__(self):
		pass 

	def policy(self,problem,state):
		# output: 
		# 	- action: [nd x 1] array 
		exit("policy not overwritten")
		
def get_solver(solver_name,
				policy_oracle=[None],
				value_oracle=None,
				search_depth=10,
				number_simulations=1000,
				C_pw=2.0,
				alpha_pw=0.5,
				C_exp=1.0,
				alpha_exp=0.25,
				beta_policy=0.0,
				beta_value=1.0,
				vis_on=False):

	if solver_name == "Empty": 
		from solvers.empty import Empty
		solver = Empty()

	elif solver_name == "MCTS": 
		from solvers.mcts import MCTS 
		solver = MCTS()		

	elif solver_name == "DARE": 
		from solvers.dare import DARE
		solver = DARE()

	elif solver_name == "PUCT_V0": 
		from solvers.puct_v0 import PUCT_V0
		solver = PUCT_V0(
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=number_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy=beta_policy,
			beta_value=beta_value,
			vis_on=vis_on
			)

	elif solver_name == "PUCT_V1": 
		from solvers.puct_v1 import PUCT_V1
		solver = PUCT_V1(
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=number_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy=beta_policy,
			beta_value=beta_value,
			vis_on=vis_on
			)

	elif solver_name == "PUCT_V2": 
		from solvers.puct_v2 import PUCT_V2
		solver = PUCT_V2(
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=number_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy=beta_policy,
			beta_value=beta_value,
			vis_on=vis_on
			)

	elif solver_name == "NeuralNetwork":
		from solvers.policy_solver import PolicySolver
		solver = PolicySolver(policy_oracle=policy_oracle)

	elif solver_name in ["C_PUCT_V0","C_PUCT_V1","C_PUCT_V2"]: 
		from solvers.c_puct import C_PUCT
		solver = C_PUCT(
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=number_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy=beta_policy,
			beta_value=beta_value,
			vis_on=vis_on,
			solver_name=solver_name
			)



	return solver 