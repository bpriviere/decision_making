

import numpy as np 
import os, subprocess
import math
import matplotlib.pyplot as plt
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


def plot_sim_result(sim_result):
	times = sim_result["times"] # nt, 
	states = sim_result["states"] # nt x state_dim
	actions = sim_result["actions"] # nt-1 x action_dim
	rewards = sim_result["rewards"] # nt-1,  

	state_dim = np.shape(states)[1]
	action_dim = np.shape(actions)[1]

	ncols = np.max((state_dim,action_dim))

	# plot trajectories (over time)
	fig,axs = plt.subplots(nrows=3,ncols=ncols)
	# state 
	for i_state in range(state_dim):
		axs[0,i_state].plot(times,states[:,i_state])
	axs[0,0].set_ylabel("States")

	# action
	for i_action in range(action_dim):
		axs[1,i_action].plot(times[1:],actions[:,i_action])
	axs[1,0].set_ylabel("Actions")

	# reward 
	axs[2,0].plot(times[1:],rewards)
	axs[2,0].set_ylabel("Rewards")

	fig.tight_layout()


def plot_tree_state(problem,tree_state):
	# tree state : nd array in [num_nodes x state_dim + 1]

	fig,ax = plt.subplots()
	position_idxs = np.arange(2)
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
	# ax.plot(nodes[:,0],nodes[:,1],'.')
	ax.scatter(nodes[0,0],nodes[0,1])

	lims = problem.S.lims
	ax.set_xlim((lims[0,0],lims[0,1]))
	ax.set_ylim((lims[1,0],lims[1,1]))
	
	# save_figs('../current/tree.pdf')
	# open_figs('../current/tree.pdf')
	# exit()