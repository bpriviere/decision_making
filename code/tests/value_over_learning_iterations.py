
import sys 
sys.path.append('../')

import matplotlib.pyplot as plt 
import numpy as np 

# custom
import plotter
from learning.oracles import get_oracles 
from problems.problem import get_problem

def eval_models(problem,model_fns,value_oracle_name,num_eval):
	
	states = []
	for _ in range(num_eval):
		state = problem.initialize()
		states.append(state)
	states = np.array(states).squeeze(axis=2)

	values = [] 
	for model_fn in model_fns:
		_, value_oracle = get_oracles(problem,
			value_oracle_name = value_oracle_name,
			value_oracle_path = model_fn
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

	plotter.save_figs("../current/plots/value_over_learning_iterations.pdf")
	plotter.open_figs("../current/plots/value_over_learning_iterations.pdf")

	# fig,ax = plt.subplots(nrows=3,ncols=3)
	# count = 0 
	# for i_row in range(3):
	# 	for i_col in range(3):

	# 		model_fn = model_fns[count]

	# 		_, value_oracle = get_oracles(problem,
	# 			value_oracle_name = value_oracle_name,
	# 			value_oracle_path = model_fn
	# 			)

	# 		values = [] 
	# 		for state in states: 
	# 			values.append(value_oracle.eval(problem,state))
	# 		values = np.array(values).squeeze(axis=2)

	# 		pos_i_idxs = np.arange(problem.state_dim)[problem.position_idx]
	# 		pcm = ax[i_row,i_col].tricontourf(states[:,pos_i_idxs[0]],states[:,pos_i_idxs[1]],values[:,0])
	# 		# fig.colorbar(pcm,ax=ax[i_row,i_col])
	# 		ax[i_row,i_col].set_xlim(problem.state_lims[problem.position_idx[0],:])
	# 		ax[i_row,i_col].set_ylim(problem.state_lims[problem.position_idx[0],:])
	# 		ax[i_row,i_col].set_title(model_fn)
	# 		problem.render(fig=fig,ax=ax[i_row,i_col])

	# 		count += 1 
	
	# fig.colorbar(pcm,ax=ax.ravel().tolist())

	# plotter.save_figs("../current/plots/value_over_learning_iterations.pdf")
	# plotter.open_figs("../current/plots/value_over_learning_iterations.pdf")


if __name__ == '__main__':
	
	problem_name = "example6"
	value_oracle_name = "deterministic"
	num_eval = 1000

	problem = get_problem(problem_name)
	model_fns = [
		"/home/ben/projects/decision_making/current/models/model_value_l{}.pt".format(i) for i in range(20)
		]

	eval_models(problem,model_fns,value_oracle_name,num_eval)