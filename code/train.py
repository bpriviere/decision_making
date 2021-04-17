
# standard 
import numpy as np 
import torch 
import random 
import pickle 
import tempfile
import itertools
import time 
import os
import multiprocessing as mp
from tqdm import tqdm 
from queue import Queue, Empty 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss

# custom
import plotter 
from param import Param 
from run import make_instance, run_instance
from solvers.c_puct import C_PUCT
from solvers.puct import PUCT
from solvers.policy_solver import PolicySolver
from learning.policy_network import PolicyNetwork
from learning.value_network import ValueNetwork
from util import write_dataset, get_dataset_fn, get_oracle_fn, format_dir

L = 2
num_D_pi = 1000
num_D_v = 10
num_self_play_plots = 10
num_simulations_expert = 10
num_simulations_learner = 10
learning_rate = 0.001
num_epochs = 10
batch_size = 2
train_test_split = 0.8
parallel_on = False
solver_name = "C_PUCT"

class Dataset(torch.utils.data.Dataset):
	
	def __init__(self, src_file, encoding_dim, target_dim, device='cpu'):
		with open(src_file, 'rb') as h:
			datapoints = np.load(h)
		self.X_np, self.target_np = datapoints[:,0:encoding_dim], datapoints[:,encoding_dim:]
		self.X_torch = torch.tensor(self.X_np,dtype=torch.float32,device=device)
		self.target_torch = torch.tensor(self.target_np,dtype=torch.float32,device=device)

	def __len__(self):
		return self.X_torch.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		return self.X_torch[idx,:], self.target_torch[idx,:]

	def to(self,device):
		self.X_torch = self.X_torch.to(device)
		self.target_torch = self.target_torch.to(device)


# utility 
def init_tqdm(rank,total):
	pbar = None 
	if rank == 0:
		pbar = tqdm(total=total)
	return pbar


def update_tqdm(rank,total_per_worker,queue,pbar):
	if rank == 0:
		count = total_per_worker
		try:
			while True:
				count += queue.get_nowait()
		except Empty:
			pass
		pbar.update(count)
	else:
		queue.put_nowait(total_per_worker)


# self play functions 
def worker_sps_wrapper(arg):
	return worker_sps(*arg)

def worker_sps(rank,queue,fn,seed,problem,num_states,total_num_states,\
	policy_oracle,value_oracle):

	np.random.seed(seed)
	pbar = init_tqdm(rank,total_num_states)
	self_play_states = [] # num sim x nt_i x state_dim
	sum_self_play_states = 0 
	while sum_self_play_states < num_states: 
		instance = dict()
		instance["problem"] = problem 
		instance["initial_state"] = problem.initialize()

		if solver_name == "PUCT":
			instance["solver"] = PUCT(\
				policy_oracle=policy_oracle,
				value_oracle=value_oracle,
				number_simulations=num_simulations_learner)
		elif solver_name == "C_PUCT":
			instance["solver"] = C_PUCT(\
				policy_oracle=policy_oracle,
				value_oracle=value_oracle,
				number_simulations=num_simulations_learner)

		sim_result = run_instance(instance,verbose=False)
		self_play_states.append(sim_result["states"])
		count_self_play_states = len(sim_result["states"])
		sum_self_play_states += count_self_play_states
		update_tqdm(rank,count_self_play_states,queue,pbar)
	np.save(fn,np.array(self_play_states))	
	del sim_result["instance"]["solver"] 
	return self_play_states

def make_self_play_states(l,robot,name,problem,num_states,policy_oracle,value_oracle):
	start_time = time.time()
	print('making self play states...')

	paths = []
	if parallel_on: 
		ncpu = mp.cpu_count() - 1
		num_states_per_pool = int(num_states/ncpu)
		seeds = [] 
		for i in range(ncpu):
			_, path = tempfile.mkstemp()
			paths.append(path + '.npy')
			seeds.append(np.random.randint(10000))
		with mp.Pool(ncpu) as pool:
			queue = mp.Manager().Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue),paths, \
				seeds, itertools.repeat(problem), itertools.repeat(num_states_per_pool), \
				itertools.repeat(num_states), itertools.repeat(policy_oracle), itertools.repeat(value_oracle)))
			for _ in pool.imap_unordered(worker_sps_wrapper, args):
				pass

	else:
		_,path = tempfile.mkstemp()
		seed = np.random.randint(10000)
		paths.append(path + '.npy')
		worker_sps_wrapper((0,Queue(),path,seed,problem,num_states,num_states,policy_oracle,value_oracle))

	self_play_states = []
	plot_count = 0 
	for path in paths:
		self_play_states_i = list(np.load(path,allow_pickle=True))
		for states in self_play_states_i:
			self_play_states.extend(states)
			if plot_count < num_self_play_plots:
				problem.render(states)
				plot_count += 1 
		os.remove(path)
	plotter.save_figs("../current/models/self_play_{}_l{}_i{}".format(name,l,robot))
	print('completed self play states in {}s.'.format(time.time()-start_time))
	return self_play_states


# expert demonstration functions 
def worker_edp_wrapper(arg):
	return worker_edp(*arg)

def worker_edp(rank,queue,fn,problem,robot,states,num_total_states,policy_oracle,value_oracle):

	# print('rank',rank)
	# print('queue',queue)
	# print('fn',fn)
	# print('problem',problem)
	# print('robot',robot)
	# # print('states',states)
	# print('num_total_states',num_total_states)
	# print('policy_oracle',policy_oracle)
	# print('value_oracle',value_oracle)
	# exit()

	pbar = init_tqdm(rank,num_total_states)
	datapoints = []
	
	if solver_name == "PUCT":
		solver = PUCT(\
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			number_simulations=num_simulations_expert)
	elif solver_name == "C_PUCT":
		solver = C_PUCT(\
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			number_simulations=num_simulations_expert)

	action_dim_per_robot = int(problem.action_dim / problem.num_robots)
	robot_action_idx = action_dim_per_robot * robot + np.arange(action_dim_per_robot)
	for state in states: 
		print('state',state)
		root_node = solver.search(problem,state)
		print('root_node.child_distribution',root_node.child_distribution)
		print('root_node.best_action',root_node.best_action)
		if root_node is not None:
			actions,num_visits = solver.get_child_distribution(root_node)
			print('actions',actions)
			encoding = problem.policy_encoding(state,robot).squeeze()
			robot_actions = np.array(actions)[:,robot_action_idx]
			target = np.average(robot_actions, weights=num_visits, axis=0)
			datapoint = np.append(encoding,target)
			datapoints.append(datapoint)
			update_tqdm(rank,1,queue,pbar)
	np.save(fn,np.array(datapoints))	
	return datapoints

def make_expert_demonstration_pi(problem,robot,states,policy_oracle,value_oracle):
	start_time = time.time()
	print('making expert demonstration pi...')

	paths = []
	if parallel_on: 
		ncpu = mp.cpu_count() - 1
		num_states_per_pool = int(len(states)/ncpu)

		paths = []
		split_states = []
		for i in range(ncpu):
			_, path = tempfile.mkstemp()
			path = path + ".npy"
			paths.append(path)
			split_states.append(states[i*num_states_per_pool:(i+1)*num_states_per_pool])

		with mp.Pool(ncpu) as pool:
			queue = mp.Manager().Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue), paths, itertools.repeat(problem), \
				itertools.repeat(robot), split_states, itertools.repeat(len(states)), itertools.repeat(policy_oracle), \
				itertools.repeat(value_oracle) ))
			for _ in pool.imap_unordered(worker_edp_wrapper, args):
				pass

	else: 
		_,path = tempfile.mkstemp()
		path = path + ".npy"
		seed = 0 
		paths.append(path)
		worker_edp_wrapper((0,Queue(),path,problem,robot,states,len(states),policy_oracle,value_oracle))

	datapoints = []
	for path in paths: 
		datapoints.extend(list(np.load(path)))
		os.remove(path)

	split = int(len(datapoints)*train_test_split)
	action_dim_per_robot = int(problem.action_dim / problem.num_robots)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_policy",\
		problem.policy_encoding_dim,action_dim_per_robot,robot=robot)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_policy",\
		problem.policy_encoding_dim,action_dim_per_robot,robot=robot)
	plotter.plot_policy_dataset(problem,train_dataset,test_dataset)
	plotter.save_figs("../current/models/dataset_policy_l{}_i{}.pdf".format(l,robot))
	print('expert demonstration pi completed in {}s.'.format(time.time()-start_time))	
	return train_dataset, test_dataset


def datapoints_to_dataset(datapoints,oracle_name,encoding_dim,target_dim,robot=0):
	dataset_fn = get_dataset_fn(oracle_name,l,robot=robot)
	datapoints = np.array(datapoints)
	write_dataset(datapoints,dataset_fn)
	dataset = Dataset(dataset_fn,encoding_dim,target_dim)
	return dataset


# value estimate 
def worker_edv_wrapper(args):
	return worker_edv(*args)


def worker_edv(fn,problem,states,policy_oracle):
	datapoints = []
	for state in states: 
		instance = dict()
		instance["problem"] = problem 
		instance["initial_state"] = state
		instance["solver"] = PolicySolver(problem,policy_oracle)
		sim_result = run_instance(instance,verbose=False)
		value = calculate_value(problem,sim_result)
		encoding = problem.value_encoding(state).squeeze()
		datapoint = np.append(encoding,value)		
		datapoints.append(datapoint)
	np.save(fn,np.array(datapoints))	
	return datapoints


def make_expert_demonstration_v(problem,states,policy_oracle):
	start_time = time.time()
	print('making expert demonstration v...')
	ncpu = mp.cpu_count() - 1
	num_states_per_pool = int(len(states)/ncpu)

	paths = []
	split_states = []
	for i in range(ncpu):
		_, path = tempfile.mkstemp()
		paths.append(path + '.npy')
		split_states.append(states[i*num_states_per_pool:(i+1)*num_states_per_pool])

	with mp.Pool(ncpu) as pool:
		args = list(zip(paths, itertools.repeat(problem), split_states, \
			itertools.repeat(policy_oracle) ))
		for _ in pool.imap_unordered(worker_edv_wrapper, args):
			pass

	datapoints = []
	for path in paths: 
		datapoints.extend(list(np.load(path)))
		os.remove(path)

	random.shuffle(datapoints) 
	split = int(len(datapoints)*train_test_split)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_value",\
		problem.value_encoding_dim,1)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_value",\
		problem.value_encoding_dim,1)
	plotter.plot_value_dataset(problem,train_dataset,test_dataset)
	plotter.save_figs("../current/models/dataset_value_l{}.pdf".format(l))
	print('expert demonstration v completed in {}s.'.format(time.time()-start_time))
	return train_dataset, test_dataset


def calculate_value(problem,sim_result):
	value = np.zeros((problem.num_robots))
	states = sim_result["states"]
	actions = sim_result["actions"]
	for step,(state,action) in enumerate(zip(states,actions)):
		reward = problem.normalized_reward(state,action)
		value += (problem.gamma ** step) * reward 
	return value 


def train_model(problem,train_dataset,test_dataset,l,oracle_name,robot=0):
	start_time = time.time()
	print('training model...')

	# device = "cpu"
	device = "cuda"
	model_fn = get_oracle_fn(oracle_name,l,robot=robot)

	if oracle_name == "policy":
		model = PolicyNetwork(problem,device=device)
	elif oracle_name == "value":
		model = ValueNetwork(problem,device=device)
	model.to(device)

	optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
	scheduler = ReduceLROnPlateau(optimizer, 'min', \
		factor=0.5, patience=50, min_lr=1e-4, verbose=True)

	train_dataset.to(device)
	test_dataset.to(device)
	train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size)
	test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size)	

	losses = []
	best_test_loss = np.Inf
	for epoch in range(num_epochs): 
		train_epoch_loss = train(model,optimizer,train_loader)
		test_epoch_loss = test(model,test_loader)
		scheduler.step(test_epoch_loss)
		losses.append((train_epoch_loss,test_epoch_loss))
		if test_epoch_loss < best_test_loss:
			best_test_loss = test_epoch_loss
			torch.save(model.to('cpu').state_dict(),model_fn)
			model.to(device)
	plotter.plot_loss(losses)
	plotter.save_figs("../current/models/losses_{}_l{}_i{}.pdf".format(oracle_name,l,robot))
	print('training model completed in {}s.'.format(time.time()-start_time))
	return 


def train(model,optimizer,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		mu, logvar = model(x,training=True)
		loss = loss_fnc(mu,logvar,target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss)
	return epoch_loss/step


def test(model,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		mu, logvar = model(x,training=True)
		loss = loss_fnc(mu,logvar,target)
		epoch_loss += float(loss)
	return epoch_loss/step


def loss_fnc(mu,logvar,target):
	criterion = MSELoss(reduction='none')
	loss = torch.sum(criterion(mu, target) / (2 * torch.exp(logvar)) + 1/2 * logvar)
	loss = loss / mu.shape[0]
	return loss 


if __name__ == '__main__':

	param = Param()
	instance = make_instance(param)
	problem = instance["problem"]
	format_dir(clean_dirnames=["data","models"]) 

	# training 
	for l in range(L):
		start_time = time.time()
		print('learning iteration: {}/{}...'.format(l,L))

		if l == 0:
			policy_oracle = [None for _ in range(problem.num_robots)]
			# policy_oracle = None 
			value_oracle = None 
		else: 
			policy_oracle = [get_oracle_fn("policy",l-1,robot=i) for i in range(problem.num_robots)]
			value_oracle = get_oracle_fn("value",l-1)

		for robot in range(problem.num_robots): 
			print('\t policy training iteration l/L, i/N: {}/{} {}/{}...'.format(\
				l,L,robot,problem.num_robots))
			states_pi = make_self_play_states(l,robot,"policy",problem,num_D_pi,policy_oracle,value_oracle)
			train_dataset_pi, test_dataset_pi = make_expert_demonstration_pi(\
				problem,robot,states_pi,policy_oracle,value_oracle)
			exit('here')
			train_model(problem,train_dataset_pi,test_dataset_pi,l,"policy",robot=robot)

		print('\t value training l/L: {}/{}'.format(l,L))
		states_v = make_self_play_states(l,robot,"value",problem,num_D_v,policy_oracle,value_oracle)
		train_dataset_v, test_dataset_v = make_expert_demonstration_v(problem,states_v,policy_oracle)
		train_model(problem,train_dataset_v,test_dataset_v,l,"value") 
		print('complete learning iteration: {}/{} in {}s'.format(l,L,time.time()-start_time))

