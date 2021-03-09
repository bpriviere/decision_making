
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
		solver = PUCT()

	elif param.solver_name == "GPUCT": 
		from solvers.game_puct import PUCT
		solver = PUCT()

	instance["param"] = param 
	instance["problem"] = problem 
	instance["solver"] = solver 
	instance["initial_state"] = problem.S.sample() 

	return instance 


def run_instance(instance):
	# input: 
	#	- 
	# outputs:
	# 	- dict of sim result 

	times, states, actions, observations, rewards = [],[],[],[],[]

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
		print('\t\t t = {}/{}'.format(step,len(problem.times)))
		
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

	print('completed sim.')

	sim_result = dict()
	sim_result["param"] = param.to_dict() 
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
	plotter.save_figs(param.curr_plot_fn)
	plotter.open_figs(param.curr_plot_fn)
