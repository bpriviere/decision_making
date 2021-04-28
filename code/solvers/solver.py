

class Solver: 

	def __init__(self):
		pass 

	def solve(self,problem,state):
		# output: 
		# 	- action: [nd x 1] array 
		exit("solve not overwritten")
		
def get_solver(param):

	if param.solver_name == "Empty": 
		from solvers.empty import Empty
		solver = Empty()

	elif param.solver_name == "MCTS": 
		from solvers.mcts import MCTS 
		solver = MCTS()		

	elif param.solver_name == "DARE": 
		from solvers.dare import DARE
		solver = DARE()

	elif param.solver_name == "PUCT_V0": 
		from solvers.puct_v0 import PUCT_V0
		solver = PUCT_V0(
			policy_oracle=param.policy_oracle,
			value_oracle=param.value_oracle,
			search_depth=param.search_depth,
			number_simulations=param.number_simulations,
			C_pw=param.C_pw,
			alpha_pw=param.alpha_pw,
			C_exp=param.C_exp,
			alpha_exp=param.alpha_exp,
			beta_policy=param.beta_policy,
			beta_value=param.beta_value,
			vis_on=param.vis_on
			)

	elif param.solver_name == "PUCT_V1": 
		from solvers.puct_v1 import PUCT_V1
		solver = PUCT_V1(
			policy_oracle=param.policy_oracle,
			value_oracle=param.value_oracle,
			search_depth=param.search_depth,
			number_simulations=param.number_simulations,
			C_pw=param.C_pw,
			alpha_pw=param.alpha_pw,
			C_exp=param.C_exp,
			alpha_exp=param.alpha_exp,
			beta_policy=param.beta_policy,
			beta_value=param.beta_value,
			vis_on=param.vis_on
			)

	elif param.solver_name == "PUCT_V2": 
		from solvers.puct_v2 import PUCT_V2
		solver = PUCT_V2(
			policy_oracle=param.policy_oracle,
			value_oracle=param.value_oracle,
			search_depth=param.search_depth,
			number_simulations=param.number_simulations,
			C_pw=param.C_pw,
			alpha_pw=param.alpha_pw,
			C_exp=param.C_exp,
			alpha_exp=param.alpha_exp,
			beta_policy=param.beta_policy,
			beta_value=param.beta_value,
			vis_on=param.vis_on
			)

	elif param.solver_name in ["C_PUCT_V0","C_PUCT_V1","C_PUCT_V2"]: 
		from solvers.c_puct import C_PUCT
		solver = C_PUCT(
			policy_oracle=param.policy_oracle,
			value_oracle=param.value_oracle,
			search_depth=param.search_depth,
			number_simulations=param.number_simulations,
			C_pw=param.C_pw,
			alpha_pw=param.alpha_pw,
			C_exp=param.C_exp,
			alpha_exp=param.alpha_exp,
			beta_policy=param.beta_policy,
			beta_value=param.beta_value,
			vis_on=param.vis_on,
			solver_name=param.solver_name
			)

	return solver 