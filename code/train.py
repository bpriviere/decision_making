
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
from problems.problem import get_problem
from solvers.solver import get_solver 
from learning.oracles import get_oracles
from run import run_instance
from util import write_dataset, get_dataset_fn, get_oracle_fn, format_dir, get_temp_fn, init_tqdm, update_tqdm

# solver 
num_simulations = 2000
search_depth = 100
C_pw = 2.0
alpha_pw = 0.5
C_exp = 1.0
alpha_exp = 0.25
beta_policy = 0.5
beta_value = 0.5
parallel_on = True
solver_name = "C_PUCT_V1"
# solver_name = "PUCT_V1"
problem_name = "example6"
policy_oracle_name = "gaussian"
value_oracle_name = "deterministic"

dirname = "../current/models"

# learning 
L = 40
mode = 1 # 0: weighted sum, 1: best child, 2: subsamples 
# num_D_pi = 10000
num_D_pi = 2000
# num_D_pi = 200
num_pi_eval = 2000
num_D_v = 10000
num_v_eval = 5000
num_subsamples = 5
num_self_play_plots = 10 
learning_rate = 0.001
num_epochs = 200
# num_epochs = 100
batch_size = 1028
train_test_split = 0.8


# MICE-like training
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

# policy demonstration functions 
def worker_edp_wrapper(arg):
	return worker_edp(*arg)

def worker_edp(rank,queue,seed,fn,problem,robot,num_per_pool,policy_oracle,value_oracle):

	np.random.seed(seed)
	pbar = init_tqdm(rank,num_D_pi)
	datapoints = []

	solver = get_solver(
			solver_name,
			policy_oracle=policy_oracle,
			value_oracle=value_oracle,
			search_depth=search_depth,
			number_simulations=num_simulations,
			C_pw=C_pw,
			alpha_pw=alpha_pw,
			C_exp=C_exp,
			alpha_exp=alpha_exp,
			beta_policy= beta_policy,
			beta_value = beta_value
			)
	
	robot_action_idx = problem.action_idxs[robot]
	count = 0
	while count < num_per_pool:
		state = problem.initialize()
		root_node = solver.search(problem,state,turn=robot)
		# print('rank: {}, completion: {}, success: {}'.format(rank,len(datapoints)/num_per_pool,root_node.success))
		if root_node.success:
			encoding = problem.policy_encoding(state,robot).squeeze()

			if mode == 0:
				# weighted average of children 
				actions,num_visits = solver.get_child_distribution(root_node)
				robot_actions = np.array(actions)[:,robot_action_idx]
				target = np.average(robot_actions, weights=num_visits, axis=0)
				datapoint = np.append(encoding,target)
				datapoints.append(datapoint)
			elif mode == 1:
				# best child 
				# most_visited_child = root_node.children[np.argmax([c.num_visits for c in root_node.children])]
				# target = root_node.edges[most_visited_child][robot_action_idx,:]

				actions,num_visits = solver.get_child_distribution(root_node)

				# print('actions',actions)
				# print('num_visits',num_visits)
				# print('np.argmax(num_visits)',np.argmax(num_visits))
				# print('actions[np.argmax(num_visits)]',actions[np.argmax(num_visits)])
				# print('actions[np.argmax(num_visits)][robot_action_idx]',actions[np.argmax(num_visits)][robot_action_idx])

				target = np.array(actions[np.argmax(num_visits)])[robot_action_idx]

				datapoint = np.append(encoding,target)
				datapoints.append(datapoint)
			elif mode == 2: 
				# subsampling of children method
				actions,num_visits = solver.get_child_distribution(root_node)
				choice_idxs = np.random.choice(len(actions),num_subsamples,p=num_visits/np.sum(num_visits))
				
				for choice_idx in choice_idxs: 
					target = np.array(actions[choice_idx])[robot_action_idx]
					datapoint = np.append(encoding,target)
					datapoints.append(datapoint)

			count += 1
			update_tqdm(rank,1,queue,pbar)
	np.save(fn,np.array(datapoints))	
	return datapoints


def make_expert_demonstration_pi(problem,robot,policy_oracle,value_oracle):
	start_time = time.time()
	print('making expert demonstration pi...')

	paths = []
	# fds = []
	if parallel_on: 
		ncpu = mp.cpu_count() - 1
		num_per_pool = int(num_D_pi / ncpu)

		seeds = [] 
		for i in range(ncpu):
			# fd, path = tempfile.mkstemp()
			# path = path + ".npy"
			# fds.append(fd)
			path = get_temp_fn(dirname,i)
			paths.append(path)
			seeds.append(np.random.randint(10000))

		with mp.Pool(ncpu) as pool:
			queue = mp.Manager().Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue), seeds, paths, itertools.repeat(problem), \
				itertools.repeat(robot), itertools.repeat(num_per_pool), itertools.repeat(policy_oracle), itertools.repeat(value_oracle)))
			for _ in pool.imap_unordered(worker_edp_wrapper, args):
				pass

	else: 
		# fd,path = tempfile.mkstemp()
		# path = path + ".npy"
		# fds.append(fd)
		path = get_temp_fn(dirname,0)
		seed = np.random.randint(10000)
		paths.append(path)
		worker_edp_wrapper((0,Queue(),seed,path,problem,robot,num_D_pi,policy_oracle,value_oracle))

	datapoints = []
	for path in paths: 
	# for fd,path in list(zip(fds,paths)): 
		# os.close(fd) 
		datapoints.extend(list(np.load(path)))
		os.remove(path)

	split = int(len(datapoints)*train_test_split)
	robot_action_dim = len(problem.action_idxs[robot])
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_policy",\
		problem.policy_encoding_dim,robot_action_dim,robot=robot)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_policy",\
		problem.policy_encoding_dim,robot_action_dim,robot=robot)
	plotter.plot_policy_dataset(problem,\
		[[train_dataset.X_np,train_dataset.target_np],[test_dataset.X_np,test_dataset.target_np]],\
		["Train","Test"],robot)
	plotter.save_figs("{}/dataset_policy_l{}_i{}.pdf".format(dirname,l,robot))
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


def worker_edv(rank,queue,fn,seed,problem,num_states_per_pool,policy_oracle):

	solver = get_solver(
			"NeuralNetwork",
			policy_oracle=policy_oracle)

	instance = {
		"problem" : problem,
		"solver" : solver,
	}
	np.random.seed(seed)
	
	count = 0 
	pbar = init_tqdm(rank,num_D_v)
	datapoints = []
	while len(datapoints) < num_states_per_pool:	
		state = problem.initialize()
		instance["initial_state"] = state
		sim_result = run_instance(0,Queue(),0,instance,verbose=False,tqdm_on=False)
		value = calculate_value(problem,sim_result)
		encoding = problem.value_encoding(state).squeeze()
		datapoint = np.append(encoding,value)
		datapoints.append(datapoint)
		update_tqdm(rank,1,queue,pbar)
	np.save(fn,np.array(datapoints))
	return datapoints


def make_expert_demonstration_v(problem, l): 
	start_time = time.time()
	print('making value dataset...')

	_, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)
	policy_oracle,_ = get_oracles(problem,
		policy_oracle_name = policy_oracle_name, 
		policy_oracle_paths = policy_oracle_paths
		)

	paths = []
	if parallel_on: 
		ncpu = mp.cpu_count() - 1
		num_states_per_pool = int(num_D_v/ncpu)
		seeds = [] 
		for i in range(ncpu):
			# _, path = tempfile.mkstemp()
			# paths.append(path + '.npy')
			paths.append(get_temp_fn(dirname,i))
			seeds.append(np.random.randint(10000))
		with mp.Pool(ncpu) as pool:
			queue = mp.Manager().Queue()
			args = list(zip(itertools.count(), itertools.repeat(queue),paths, \
				seeds, itertools.repeat(problem), itertools.repeat(num_states_per_pool), \
				itertools.repeat(policy_oracle) ))

			for _ in pool.imap_unordered(worker_edv_wrapper, args):
				pass

	else:
		# _,path = tempfile.mkstemp()
		# paths.append(path + '.npy')
		paths = [get_temp_fn(dirname,0)]
		seed = np.random.randint(10000)
		worker_edv_wrapper((0,Queue(),paths[0],seed,problem,num_D_v,policy_oracle))

	datapoints = []
	plot_count = 0 
	for path in paths:
		datapoints_i = np.load(path,allow_pickle=True)
		datapoints.extend(datapoints_i)
		os.remove(path)

	split = int(len(datapoints)*train_test_split)
	train_dataset = datapoints_to_dataset(datapoints[0:split],"train_value",\
		problem.value_encoding_dim,problem.num_robots)
	test_dataset = datapoints_to_dataset(datapoints[split:],"test_value",\
		problem.value_encoding_dim,problem.num_robots)
	plotter.plot_value_dataset(problem,
		[[train_dataset.X_np,train_dataset.target_np],[test_dataset.X_np,test_dataset.target_np]],
		["Train","Test"])
	plotter.save_figs("{}/dataset_value_l{}.pdf".format(dirname,l))
	print('expert demonstration v completed in {}s.'.format(time.time()-start_time))	
	return train_dataset, test_dataset


def calculate_value(problem,sim_result):
	value = np.zeros((problem.num_robots,1))
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
	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)

	if oracle_name == "policy":
		model_fn = policy_oracle_paths[robot]
		model, _ = get_oracles(problem,
			policy_oracle_name = policy_oracle_name,
			policy_oracle_paths = [None for _ in range(problem.num_robots)],
			force = True
			)
		model = model[robot]
	elif oracle_name == "value":
		model_fn = value_oracle_path
		_, model = get_oracles(problem,
			value_oracle_name = value_oracle_name,
			force = True
			)
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
	for epoch in tqdm(range(num_epochs)): 
		train_epoch_loss = train(model,optimizer,train_loader)
		test_epoch_loss = test(model,test_loader)
		scheduler.step(test_epoch_loss)
		losses.append((train_epoch_loss,test_epoch_loss))
		if test_epoch_loss < best_test_loss:
			best_test_loss = test_epoch_loss
			torch.save(model.to('cpu').state_dict(),model_fn)
			model.to(device)
	plotter.plot_loss(losses)
	plotter.save_figs("{}/losses_{}_l{}_i{}.pdf".format(dirname,oracle_name,l,robot))
	print('training model completed in {}s.'.format(time.time()-start_time))
	return 


def train(model,optimizer,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		loss = model.loss_fnc(x,target) 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		epoch_loss += float(loss)
	return epoch_loss/step


def test(model,loader):
	epoch_loss = 0
	loss_by_components = []
	for step, (x,target) in enumerate(loader):
		loss = model.loss_fnc(x,target) 
		epoch_loss += float(loss)
	return epoch_loss/step


def eval_value(problem,l):
	
	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)

	_, value_oracle = get_oracles(problem,
		value_oracle_name = value_oracle_name,
		value_oracle_path = value_oracle_path
		)

	states = []
	values = []
	encodings = [] 
	for _ in range(num_v_eval):
		state = problem.initialize()
		encoding = problem.value_encoding(state)
		value = value_oracle.eval(problem,state)
		states.append(state)
		values.append(value)
		encodings.append(encoding.reshape((problem.value_encoding_dim,1)))

	states = np.array(states).squeeze(axis=2)
	values = np.array(values).squeeze(axis=2)
	encodings = np.array(encodings).squeeze(axis=2)
	plotter.plot_value_dataset(problem,[[encodings,values]],["Eval"])
	# plotter.plot_value_dataset(problem,[[states,values]],["Eval"])
	plotter.save_figs("{}/value_eval_l{}.pdf".format(dirname,l))


def eval_policy(problem,l,robot):

	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)
	for robot_i,path in enumerate(policy_oracle_paths):
		if robot_i != robot:
			policy_oracle_paths[robot_i] = None
	
	policy_oracles, _ = get_oracles(problem,
		policy_oracle_name = policy_oracle_name,
		policy_oracle_paths = policy_oracle_paths
		)
	policy_oracle = policy_oracles[robot]

	states = []
	encodings = []
	actions = []
	robot_action_dim = len(problem.action_idxs[robot]) 
	for _ in range(num_pi_eval):
		state = problem.initialize()
		encoding = problem.policy_encoding(state,robot)
		encoding = torch.tensor(encoding,dtype=torch.float32).squeeze().unsqueeze(0) # [batch_size x state_dim]
		mu, logvar = policy_oracle(encoding,training=True) # mu in [1 x robot_action_dim]
		mu = mu.detach().numpy().reshape((robot_action_dim,1))
		sd = np.sqrt(np.exp(logvar.detach().numpy().reshape((robot_action_dim,1))))
		action = np.concatenate((mu,sd),axis=0)
		states.append(state)
		actions.append(action)
		encodings.append(encoding.detach().numpy().reshape((problem.policy_encoding_dim,1)))

	states = np.array(states).squeeze(axis=2)
	actions = np.array(actions).squeeze(axis=2)
	encodings = np.array(encodings).squeeze(axis=2)
	# plotter.plot_policy_dataset(problem,[[states,actions]],["Eval"],robot)
	plotter.plot_policy_dataset(problem,[[encodings,actions]],["Eval"],robot)
	plotter.save_figs("{}/policy_eval_l{}_i{}.pdf".format(dirname,l,robot))


def self_play(problem,policy_oracle,value_oracle,l):
	solver = get_solver(
			"NeuralNetwork",
			policy_oracle=policy_oracle)

	instance = {
		"problem" : problem,
		"solver" : solver,
		"policy_oracle" : policy_oracle,
		"value_oracle" : value_oracle,
	}

	sim_results = [] 
	for _ in range(num_self_play_plots):
		state = problem.initialize()
		instance["initial_state"] = state
		sim_result = run_instance(0,Queue(),0,instance,verbose=False,tqdm_on=False)
		sim_results.append(sim_result)

	# if parallel_on:
	# 	pool = mp.Pool(mp.cpu_count() - 1)
	# 	params = [Param() for _ in range(num_self_play_plots)]
	# 	seeds = [np.random.randint(10000) for _ in range(num_self_play_plots)]
	# 	args = list(zip(
	# 		itertools.count(), 
	# 		itertools.repeat(mp.Manager().Queue()),
	# 		itertools.repeat(param.num_self_play_plots),
	# 		params,seeds))
	# 	sim_results = pool.imap_unordered(_worker_run_instance, args)
	# 	pool.close()
	# 	pool.join()
	# else:
	# 	sim_results = [run_instance(0,Queue(),len(instance["problem"].times),instance,verbose=False,tqdm_on=True)]

	for sim_result in sim_results:
		plotter.plot_sim_result(sim_result)
		problem.render(states=sim_result["states"])

	if hasattr(problem, 'pretty_plot'):
		problem.pretty_plot(sim_results[0])	
	
	plotter.save_figs("{}/self_play_l{}.pdf".format(dirname,l))
	return sim_results


if __name__ == '__main__':

	problem = get_problem(problem_name) 
	format_dir(clean_dirnames=["data","models"]) 

	num_D_pi_samples = num_D_pi
	if mode == 2:
		num_D_pi_samples = num_D_pi*num_subsamples
	if batch_size > np.min((num_D_pi_samples,num_D_v)) * (1-train_test_split):
		batch_size = int(np.floor((np.min((num_D_pi_samples,num_D_v)) * train_test_split / 10)))
		print('changing batch size to {}'.format(batch_size))

	# training 
	for l in range(L):
		start_time = time.time()
		print('learning iteration: {}/{}...'.format(l,L))

		if l == 0:
			policy_oracle = [None for _ in range(problem.num_robots)]
			value_oracle = None
		else: 
			value_oracle_path, policy_oracle_paths = get_oracle_fn(l-1,problem.num_robots)
			policy_oracle,value_oracle = get_oracles(problem,
				value_oracle_name = value_oracle_name,
				value_oracle_path = value_oracle_path,
				policy_oracle_name = policy_oracle_name, 
				policy_oracle_paths = policy_oracle_paths
				)
			
			print('\t self play l/L: {}/{}...'.format(l,L))
			sim_results = self_play(problem,policy_oracle,value_oracle,l-1)


		for robot in range(problem.num_robots): 
			print('\t policy training iteration l/L, i/N: {}/{} {}/{}...'.format(\
				l,L,robot,problem.num_robots))
			train_dataset_pi, test_dataset_pi = make_expert_demonstration_pi(\
				problem,robot,policy_oracle,value_oracle)
			train_model(problem,train_dataset_pi,test_dataset_pi,l,"policy",robot=robot)
			eval_policy(problem,l,robot) 

		print('\t value training l/L: {}/{}'.format(l,L))
		train_dataset_v, test_dataset_v = make_expert_demonstration_v(problem, l) 
		train_model(problem,train_dataset_v,test_dataset_v,l,"value") 
		eval_value(problem,l)
		print('complete learning iteration: {}/{} in {}s'.format(l,L,time.time()-start_time))
