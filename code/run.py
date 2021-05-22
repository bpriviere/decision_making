
# standard 
import numpy as np 
import multiprocessing as mp
import itertools
from queue import Queue

# custom 
from param import Param 
from problems.problem import get_problem
from solvers.solver import get_solver 
from learning.oracles import get_oracles 
import plotter 
from util import init_tqdm, update_tqdm


def make_instance(param):

	instance = dict() 

	problem = get_problem(param.problem_name)
	policy_oracle,value_oracle = get_oracles(problem,
		value_oracle_name = param.value_oracle_name,
		value_oracle_path = param.value_oracle_path,
		policy_oracle_name = param.policy_oracle_name,
		policy_oracle_paths = param.policy_oracle_paths
		)
	solver = get_solver(param.solver_name,
		policy_oracle=policy_oracle,
		value_oracle=value_oracle,
		search_depth=param.search_depth,
		number_simulations=param.number_simulations,
		C_pw=param.C_pw,
		alpha_pw=param.alpha_pw,
		C_exp=param.C_exp,
		alpha_exp=param.alpha_exp,
		beta_policy=param.beta_policy,
		beta_value=param.beta_value,
		vis_on=param.vis_on)

	instance["policy_oracle"] = policy_oracle
	instance["value_oracle"] = value_oracle
	instance["problem"] = problem 
	instance["solver"] = solver 
	instance["initial_state"] = problem.initialize()

	# instance["initial_state"] = np.array([
	# 	# [-1],[3], # state for single robot, 2d single integrator problems
	# 	# [1],[1],[1],[-2],[0],[0], # state for homicidal chauffeur 
	# 	[1],[1],[-2],[1],[np.pi],[0], # state for homicidal chauffeur 
	# 	])

	return instance 


def run_instance(rank,queue,total,instance,verbose=False,tqdm_on=True):
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

	# print('rank',rank)
	# print('total',total)
	if tqdm_on:	pbar = init_tqdm(rank,total)

	states.append(curr_state)
	times.append(problem.times[0])
	for step,time in enumerate(problem.times[1:]):

		if verbose and not tqdm_on: print('\t\t t = {}/{}'.format(step,len(problem.times)))
		
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

		if tqdm_on: update_tqdm(rank,1,queue,pbar)

		if done: 
			break 
		else: 
			curr_state = next_state

	if verbose:	print('completed sim.')
	if verbose: problem.render(states=np.array(states))

	sim_result = dict()
	sim_result["instance"] = instance
	sim_result["times"] = times 
	sim_result["states"] = np.array(states)
	sim_result["actions"] = np.array(actions)
	sim_result["rewards"] = np.array(rewards)

	return sim_result

def worker_run_instance(rank,queue,num_trials,param,seed):
	np.random.seed(seed)
	instance = make_instance(param)
	total = num_trials * len(instance["problem"].times)
	sim_result = run_instance(rank,queue,total,instance)
	del sim_result["instance"]["solver"] # can't pickle bindings 
	return sim_result

def _worker_run_instance(arg):
	return worker_run_instance(*arg)


if __name__ == '__main__':

	param = Param()

	print('running sim...')
	if param.parallel_on:
		pool = mp.Pool(mp.cpu_count() - 1)
		params = [Param() for _ in range(param.num_trials)]
		seeds = [np.random.randint(10000) for _ in range(param.num_trials)]
		args = list(zip(
			itertools.count(), 
			itertools.repeat(mp.Manager().Queue()),
			itertools.repeat(param.num_trials),
			params,seeds))
		sim_results = pool.imap_unordered(_worker_run_instance, args)
		# sim_results = pool.map(_worker, args)
		pool.close()
		pool.join()
	else:
		instance = make_instance(param)
		sim_results = [run_instance(0,Queue(),len(instance["problem"].times),instance,verbose=True)]

	if param.movie_on: 
		print('making movie...')
		plotter.make_movie(sim_results[0],sim_result[0]["instance"],"../current/plots/vid.mp4")
		plotter.open_figs("../current/plots/vid.mp4")	

	# save/load results
	# todo

	# plotting 
	print('plotting results...')
	for sim_result in sim_results:
		plotter.plot_sim_result(sim_result)
		sim_result["instance"]["problem"].render(states=sim_result["states"])
		if param.pretty_plot_on and hasattr(sim_result["instance"]["problem"], 'pretty_plot') :
			sim_result["instance"]["problem"].pretty_plot(sim_result)

	plotter.save_figs("../current/plots/run.pdf")
	plotter.open_figs("../current/plots/run.pdf")