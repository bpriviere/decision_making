
# standard 
import numpy as np 

# custom 
from param import Param 
from problems.problem import get_problem
from solvers.solver import get_solver 
import plotter 
import util


def make_instance(param):

	instance = dict() 

	problem = get_problem(param)
	solver = get_solver(param)

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

		dt = problem.dt 
		if solver.solver_name in ["PUCT_V2","C_PUCT_V2"]:
			dt = action[-1,0]
			action = action[0:-1,:]

		reward = problem.reward(curr_state,action)
		next_state = problem.step(curr_state,action,dt)
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
	sim_result["rewards"] = np.array(rewards)

	return sim_result


if __name__ == '__main__':

	param = Param()

	# make instance 
	instance = make_instance(param)

	# # run instance 
	sim_result = run_instance(instance)

	if param.movie_on: 
		print('making movie...')
		plotter.make_movie(sim_result,instance,"../current/plots/vid.mp4")
		plotter.open_figs("../current/plots/vid.mp4")	

	# save/load results
	# todo

	# plotting 
	print('plotting results...')
	plotter.plot_sim_result(sim_result)
	plotter.save_figs("../current/plots/run.pdf")
	plotter.open_figs("../current/plots/run.pdf")