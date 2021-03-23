
# standard 
import numpy as np 

# custom 
from param import Param 
import plotter 
import util


def make_instance(param):

	instance = dict() 

	if param.problem_name == "example1":
		from problems.example1 import Example1
		problem = Example1()

	elif param.problem_name == "example2":
		from problems.example2 import Example2
		problem = Example2()

	if param.solver_name == "Empty": 
		from solvers.empty import Empty
		solver = Empty()

	elif param.solver_name == "MCTS": 
		from solvers.mcts import MCTS 
		solver = MCTS()		

	elif param.solver_name == "DARE": 
		from solvers.dare import DARE
		solver = DARE()

	elif param.solver_name == "PUCT": 
		from solvers.puct import PUCT
		solver = PUCT(vis_on=param.tree_vis_on)

	elif param.solver_name == "C_PUCT": 
		from solvers.c_puct import C_PUCT
		solver = C_PUCT()

	instance["problem"] = problem 
	instance["solver"] = solver 
	instance["initial_state"] = problem.initialize() 

	return instance 


def run_instance(instance,verbose=True):
	# input: 
	#	- 
	# outputs:
	# 	- dict of sim result 

	times, states, actions, observations, rewards = [],[],[],[],[]

	if verbose:
		print('   running sim with... \n\t{} \n\t{} \n\t{}'.format(\
			instance["problem"],
			instance["solver"],
			instance["initial_state"]
			))

	problem = instance["problem"] 
	solver = instance["solver"] 
	curr_state = instance["initial_state"]

	states.append(curr_state)
	times.append(problem.times[0])
	for step,time in enumerate(problem.times[1:]):

		if verbose: print('\t\t t = {}/{}'.format(step,len(problem.times)))
		
		action = solver.policy(problem,curr_state)
		reward = problem.reward(curr_state,action)
		next_state = problem.step(curr_state,action)
		done = problem.is_terminal(next_state)

		times.append(time)
		states.append(next_state)
		actions.append(action)
		rewards.append(reward)

		if done: 
			break 
		else: 
			curr_state = next_state

	if verbose:	print('completed sim.')
	if verbose: problem.render(np.array(states))

	sim_result = dict()
	sim_result["instance"] = instance
	sim_result["times"] = times 
	sim_result["states"] = np.array(states)
	sim_result["actions"] = np.array(actions)
	sim_result["rewards"] = np.array(rewards).squeeze()

	return sim_result


if __name__ == '__main__':

	param = Param()

	# make instance 
	instance = make_instance(param)

	# run instance 
	sim_result = run_instance(instance)

	# save/load results
	# todo

	# plotting 
	print('plotting results...')
	plotter.plot_sim_result(sim_result)
	plotter.save_figs("../current/plots/run.pdf")
	plotter.open_figs("../current/plots/run.pdf")
