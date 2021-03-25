
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
		problem = Example1(
			t0 = param.t0,
			tf = param.tf,
			dt = param.dt,
			pos_lim = param.pos_lim,
			vel_lim = param.vel_lim)

	elif param.problem_name == "example2":
		from problems.example2 import Example2
		problem = Example2(
			t0 = param.t0,
			tf = param.tf,
			dt = param.dt,
			pos_lim = param.pos_lim,
			vel_lim = param.vel_lim,
			acc_lim = param.acc_lim,
			mass = param.mass)

	elif param.problem_name == "example3":
		from problems.example3 import Example3
		problem = Example3(
			t0 = param.t0,
			tf = param.tf,
			dt = param.dt,
			pos_lim = param.pos_lim,
			vel_lim = param.vel_lim,
			acc_lim = param.acc_lim,
			rad_lim = param.rad_lim,
			omega_lim = param.omega_lim,
			desired_distance = param.desired_distance,
			state_control_weight = param.state_control_weight,
			g = param.g)

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
		solver = PUCT(
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

	elif param.solver_name == "C_PUCT": 
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
			vis_on=param.vis_on
			)

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
