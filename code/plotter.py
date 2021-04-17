

import numpy as np 
import os, subprocess
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import cm	
from matplotlib.backends.backend_pdf import PdfPages 
from PyPDF2 import PdfFileMerger

# defaults
plt.rcParams.update({'font.size': 10})
plt.rcParams['lines.linewidth'] = 2.5

import matplotlib
matplotlib.use('Agg')


def has_figs():
	if len(plt.get_fignums()) > 0:
		return True
	else:
		return False


def save_figs(filename):
	file_dir,  file_name = os.path.split(filename)
	if len(file_dir) >0 and not (os.path.isdir(file_dir)):
		os.makedirs(file_dir)
	fn = os.path.join( os.getcwd(), filename)
	pp = PdfPages(fn)
	for i in plt.get_fignums():
		pp.savefig(plt.figure(i))
		plt.close(plt.figure(i))
	pp.close()


def open_figs(filename):
	pdf_path = os.path.join( os.getcwd(), filename)
	if os.path.exists(pdf_path):
		subprocess.call(["xdg-open", pdf_path])


def merge_figs(pdfs,result_fn):
	merger = PdfFileMerger()
	# write new one 
	for pdf in pdfs:
	    merger.append(pdf)
	merger.write(result_fn)
	merger.close()
	# delete old files 
	for pdf in pdfs: 
		os.remove(pdf)


def make_fig():
	return plt.subplots()


def make_3d_fig():
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	return fig,ax 


def get_n_colors(n,cmap=None):
	colors = []
	cm_subsection = np.linspace(0, 1, n)
	if cmap is None:
		cmap = cm.tab20
	colors = [ cmap(x) for x in cm_subsection]
	return colors


def plot_sim_result(sim_result):
	times = sim_result["times"] # nt, 
	states = sim_result["states"] # nt x state_dim
	actions = sim_result["actions"] # nt-1 x action_dim
	rewards = sim_result["rewards"] # nt-1,  
	problem = sim_result["instance"]["problem"] 
	problem = problem.__dict__ 

	num_robots = problem["num_robots"]
	state_dim_per_robot = int(np.shape(states)[1] / num_robots)
	action_dim = np.shape(actions)[1]
	state_lims = problem["state_lims"]
	action_lims = problem["action_lims"]

	ncols = np.max((state_dim_per_robot,action_dim))

	# plot trajectories (over time)
	fig,axs = plt.subplots(nrows=int(num_robots+2),ncols=int(ncols))
	# state 
	for i_robot in range(num_robots):
		for i_state in range(state_dim_per_robot):
			idx = i_state + state_dim_per_robot * i_robot 
			axs[i_robot,i_state].plot(times,states[:,idx])
			axs[i_robot,i_state].set_ylim((state_lims[idx,0],state_lims[idx,1]))
		axs[i_robot,0].set_ylabel("Robot State {}".format(i_robot))

	# action
	for i_action in range(action_dim):
		axs[num_robots,i_action].plot(times[1:],actions[:,i_action])
		axs[num_robots,i_action].set_ylim((action_lims[i_action,0],action_lims[i_action,1]))
	axs[num_robots,0].set_ylabel("Actions")

	# reward 
	for i_robot in range(num_robots):
		axs[num_robots+1,0].plot(times[1:],rewards[:,i_robot])
	axs[num_robots+1,0].set_ylabel("Rewards")

	# fig.tight_layout()


def plot_loss(losses):
	fig,ax = plt.subplots()
	ax.plot(losses)
	ax.set_title("Losses")


def plot_tree_state(problem,tree_state,zoom_on=True):
	# tree state : nd array in [num_nodes x state_dim + 1]

	position_idxs = problem.position_idx

	if len(position_idxs) == 2: 
		fig,ax = plt.subplots()
		segments = []
		nodes = [] 
		for i_row,row in enumerate(tree_state):
			parentIdx = int(row[-1])
			nodes.append(row[position_idxs])
			if parentIdx >= 0:
				segments.append([row[position_idxs], tree_state[parentIdx][position_idxs]])

		ln_coll = matplotlib.collections.LineCollection(segments, linewidth=0.2, colors='k', alpha=0.2)
		nodes = np.array(nodes)

		ax.add_collection(ln_coll)
		ax.scatter(nodes[0,0],nodes[0,1])

		if not zoom_on: 
			lims = problem.state_lims
			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))

	elif len(position_idxs) == 3: 
		
		num_robots = problem.num_robots
		state_dim_per_robot = int(problem.state_dim / num_robots)

		fig,ax = make_3d_fig()
		segments = [[] for _ in range(num_robots)]
		nodes = [[] for _ in range(num_robots)]
		for i_row,row in enumerate(tree_state):
			parentIdx = int(row[-1])

			for robot in range(num_robots):
				robot_state_idx = robot * state_dim_per_robot + np.arange(state_dim_per_robot)
				robot_position_idx = robot_state_idx[position_idxs]
				nodes[robot].append(row[robot_position_idx])
				if parentIdx >= 0:
					segments[robot].append([row[robot_position_idx], tree_state[parentIdx][robot_position_idx]])

		for robot in range(num_robots):
			ln_coll = Line3DCollection(segments[robot], linewidth=0.2, colors='k', alpha=0.2)
			ax.add_collection(ln_coll)

		if not zoom_on: 
			lims = problem.state_lims
			ax.set_xlim((lims[0,0],lims[0,1]))
			ax.set_ylim((lims[1,0],lims[1,1]))
			ax.set_zlim((lims[2,0],lims[2,1]))

	else: 
		print('tree plot dimension not supported')


def plot_value_dataset(problem,train_dataset,test_dataset):
	
	encoding_dim = problem.policy_encoding_dim
	target_dim = 1
	state_lims = problem.state_lims
	# action_lims = [0,1]

	for title,dataset in zip(["Train","Test"],[train_dataset,test_dataset]):
		encodings = dataset.X_np 
		target = dataset.target_np 

		fig,ax = plt.subplots(nrows=2,ncols=max((encoding_dim,target_dim)),squeeze=False)
		for i_e in range(encoding_dim):
			ax[0,i_e].hist(encodings[:,i_e])
			ax[0,i_e].set_xlim(state_lims[i_e,0],state_lims[i_e,1])
		ax[0,0].set_ylabel("Encoding")

		for i_t in range(target_dim):
			ax[1,i_t].hist(target[:,i_t])
			# ax[1,i_t].set_xlim(action_lims[i_t,0],action_lims[i_t,1])
		ax[1,0].set_ylabel("Target")
		fig.suptitle(title)


def plot_policy_dataset(problem,train_dataset,test_dataset):
	# datapoints: [(encoding,target) ]
	# encoding: problem.policy_encoding(state)
	# target: robot_action 

	encoding_dim = problem.policy_encoding_dim
	target_dim = int(problem.action_dim / problem.num_robots)
	state_lims = problem.state_lims
	action_lims = problem.action_lims

	for title,dataset in zip(["Train","Test"],[train_dataset,test_dataset]):
		encodings = dataset.X_np 
		target = dataset.target_np 

		fig,ax = plt.subplots(nrows=2,ncols=max((encoding_dim,target_dim)),squeeze=False)
		for i_e in range(encoding_dim):
			ax[0,i_e].hist(encodings[:,i_e])
			ax[0,i_e].set_xlim(state_lims[i_e,0],state_lims[i_e,1])
		ax[0,0].set_ylabel("Encoding")

		for i_t in range(target_dim):
			ax[1,i_t].hist(target[:,i_t])
			ax[1,i_t].set_xlim(action_lims[i_t,0],action_lims[i_t,1])
		ax[1,0].set_ylabel("Target")
		fig.suptitle(title)


def plot_regression_test(results):
	# results: list of (instance, sim_result) pairs 

	# render each sim result 
	for (param, sim_result) in results:
		# fig,ax = sim_result["instance"]["problem"].render(sim_result["states"])
		# fig.suptitle(param.key)
		pass 

	# for each problem, 
	# 	- plot duration per timestep across number of simulations for each solver 
	# 	- plot total reward ""
	duration_plot_data = np.zeros((
		len(param.problem_name_lst),
		len(param.number_simulations_lst),
		len(param.solver_name_lst),
		param.num_trial))
	reward_plot_data = np.zeros((
		len(param.problem_name_lst),
		len(param.number_simulations_lst),
		len(param.solver_name_lst),
		param.num_trial))
	for (param,sim_result) in results: 
		i_pn = param.problem_name_lst.index(param.problem_name)
		i_ns = param.number_simulations_lst.index(param.number_simulations)
		i_sn = param.solver_name_lst.index(param.solver_name)
		duration_plot_data[i_pn,i_ns,i_sn,param.trial] = sim_result["duration_per_timestep"]
		reward_plot_data[i_pn,i_ns,i_sn,param.trial] = np.sum(sim_result["rewards"][:,0])
	
	fig,axs = plt.subplots(ncols=len(param.problem_name_lst),nrows=1,squeeze=False)
	for i_pn, problem_name in enumerate(param.problem_name_lst):
		for i_sn, solver_name in enumerate(param.solver_name_lst):
			lns = axs[0,i_pn].plot(
				np.array(param.number_simulations_lst),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				label = solver_name)
			axs[0,i_pn].fill_between(
				np.array(param.number_simulations_lst),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1) - np.std(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				np.mean(duration_plot_data[i_pn,:,i_sn,:],axis=1) + np.std(duration_plot_data[i_pn,:,i_sn,:],axis=1),
				color = lns[0].get_color(),
				alpha = 0.5)
		axs[0,i_pn].set_xscale('log')
		axs[0,i_pn].set_yscale('log')
		axs[0,i_pn].set_title(problem_name)
		axs[0,i_pn].tick_params(axis='x', rotation=45)
	axs[0,0].legend(loc='best')
	axs[0,0].set_ylabel("WCT / Timestep")
	fig.suptitle("Computation Time Regression Test")

	fig,axs = plt.subplots(ncols=len(param.problem_name_lst),nrows=1,squeeze=False)
	for i_pn, problem_name in enumerate(param.problem_name_lst):
		for i_sn, solver_name in enumerate(param.solver_name_lst):
			lns = axs[0,i_pn].plot(
				np.array(param.number_simulations_lst),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				label = solver_name)
			axs[0,i_pn].fill_between(
				np.array(param.number_simulations_lst),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1) - np.std(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				np.mean(reward_plot_data[i_pn,:,i_sn,:],axis=1) + np.std(reward_plot_data[i_pn,:,i_sn,:],axis=1),
				color = lns[0].get_color(),
				alpha = 0.5)
		axs[0,i_pn].set_xscale('log')
		axs[0,i_pn].set_title(problem_name)
		axs[0,i_pn].tick_params(axis='x', rotation=45)
	axs[0,0].legend(loc='best')
	axs[0,0].set_ylabel("Total Reward")
	fig.suptitle("Total Reward Regression Test")


def make_movie(sim_result,instance,filename):

	from matplotlib import animation

	states = sim_result["states"]
	times = sim_result["times"] 

	if instance["problem"].name in ["example3","example4"]:
		fig,ax = make_3d_fig()
	else:
		fig,ax = make_fig()

	def init(): 
		ax.clear()
		ax.grid(True)

	# animate over trajectory
	def animate(i_t):
		init()
		print(i_t/len(times))
		time_idxs = range(i_t) #times[0:i_t]
		states_i = states[time_idxs]
		if i_t < 2:
			return ln 
		else:
			instance["problem"].render(states_i,fig=fig,ax=ax)
		return ln 

	ln = ax.plot([],[],[])
	anim = animation.FuncAnimation(fig, animate, frames=len(times)+1, interval=1)
	anim.save(filename, fps=10, extra_args=['-vcodec', 'libx264'], dpi=1000)