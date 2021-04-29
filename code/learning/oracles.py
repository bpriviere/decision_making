

def get_oracles(problem, 
	policy_oracle_name = None, 
	policy_oracle_path = None, 
	value_oracle_name = None,
	value_oracle_path = None,
	force=False):

	policy_oracle = [None for _ in range(problem.num_robots)]
	value_oracle = None 

	if value_oracle_name == "deterministic":
		if value_oracle_path is not None or force:
			from learning.deterministic_value_network import DeterministicValueNetwork
			value_oracle = DeterministicValueNetwork(problem,path=value_oracle_path)

	elif value_oracle_name == "gaussian":
		if value_oracle_path is not None or force:
			from learning.gaussian_value_network import GaussianValueNetwork
			value_oracle = GaussianValueNetwork(problem,path=value_oracle_path)

	else: 
		print("value_network not recognized: {}".format(value_network))
		exit()

	return policy_oracle, value_oracle  