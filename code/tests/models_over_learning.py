
import sys 
sys.path.append('../')

import matplotlib.pyplot as plt 
import numpy as np 

# custom
import plotter
from learning.oracles import get_oracles 
from problems.problem import get_problem

def eval_models(problem,policy_model_fns,policy_oracle_name,value_model_fns,value_oracle_name,num_eval,num_robots):
	
	states = []
	for _ in range(num_eval):
		state = problem.initialize()
		states.append(state)
	states = np.array(states).squeeze(axis=2)


	# ------------- value ---------------
	values = [] 
	for value_model_fn in value_model_fns:
		_, value_oracle = get_oracles(problem,
			value_oracle_name = value_oracle_name,
			value_oracle_path = value_model_fn,
			# policy_oracle_name = policy_oracle_name, 
			# policy_oracle_paths = policy_model_fns,
			)

		values_i = [] 
		for state in states: 
			values_i.append(value_oracle.eval(problem,state))
		values_i = np.array(values_i).squeeze(axis=2)
		values.append(values_i)

	# normalize
	values = np.array(values)
	min_value = np.amin(values)
	max_value = np.amax(values)
	values = (values - min_value) / (max_value - min_value)

	# plot 
	for i_model in range(values.shape[0]):
		values_i = values[i_model,:,:]

		fig,ax = plt.subplots()
		pos_i_idxs = np.arange(problem.state_dim)[problem.position_idx]
		pcm = ax.tricontourf(states[:,pos_i_idxs[0]],states[:,pos_i_idxs[1]],values_i[:,0],vmin=0,vmax=1)
		fig.colorbar(pcm,ax=ax)
		ax.set_xlim(problem.state_lims[problem.position_idx[0],:])
		ax.set_ylim(problem.state_lims[problem.position_idx[0],:])
		ax.set_title(i_model)
		problem.render(fig=fig,ax=ax)


	# ------------- policy ---------------
	# todo 


	plotter.save_figs("../current/plots/model_over_learning_iterations.pdf")
	plotter.open_figs("../current/plots/model_over_learning_iterations.pdf")

if __name__ == '__main__':
	
	problem_name = "example6"
	value_oracle_name = "deterministic"
	policy_oracle_name = "gaussian"
	num_eval = 1000

	num_iterations = 20
	num_robots = 2

	problem = get_problem(problem_name)
	value_model_fns = [
		"/home/ben/projects/decision_making/current/models/model_value_l{}.pt".format(l) for l in range(num_iterations)
		]

	policy_model_fns = []
	for i in range(num_robots):
		policy_model_fns.append(["/home/ben/projects/decision_making/current/models/model_policy_l{}_i{}.pt".format(l,i) for l in range(num_iterations)])

	eval_models(problem,policy_model_fns,policy_oracle_name,value_model_fns,value_oracle_name,num_eval,num_robots)