
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
from util import write_dataset, get_dataset_fn, get_oracle_fn, format_dir


parallel_on = True

# solver settings 
# number_simulations = 50
number_simulations = 500
search_depth = 50
C_pw = 10.0
alpha_pw = 0.25
C_exp = 10.0
alpha_exp = 0.5
beta_policy = 0.0
beta_value = 0.7
vis_on = True

solver_name = "C_PUCT_V1"
# solver_name = "PUCT_V1"
problem_name = "example6"
value_oracle_name = "deterministic"

# learning 
L = 20
num_D_v = 2000
# num_D_v = 200
num_v_eval = 2000

learning_rate = 0.001
num_epochs = 1000
# num_epochs = 100
batch_size = 128
train_test_split = 0.8


# Shah-like training 
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


# expert value demonstration functions 
def worker_edv_wrapper(arg):
	return worker_edv(*arg)


def worker_edv(rank,queue,fn,seed,problem,num_states,total_num_states,\
	policy_oracle,value_oracle):

	np.random.seed(seed)
	pbar = init_tqdm(rank,total_num_states)
	datapoints = [] 
	while len(datapoints) < num_states:
		solver = get_solver(
				solver_name,
				policy_oracle=policy_oracle,
				value_oracle=value_oracle,
				number_simulations=number_simulations,
				beta_value = beta_value)
		root_state = problem.sample_state()
		root_node = solver.search(problem,root_state)
		if root_node.success:
			datapoint = np.append(root_state.squeeze(),root_node.value)
			datapoints.append(datapoint)
			update_tqdm(rank,1,queue,pbar)
	datapoints = np.array(datapoints)
	np.save(fn,datapoints)	
	del solver 
	return datapoints


def make_expert_demonstration_v(problem, num_states, value_oracle, policy_oracle): 
	start_time = time.time()
	print('making value dataset...')

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
			for _ in pool.imap_unordered(worker_edv_wrapper, args):
				pass

	else:
		_,path = tempfile.mkstemp()
		seed = np.random.randint(10000)
		paths.append(path + '.npy')
		worker_edv_wrapper((0,Queue(),path,seed,problem,num_states,num_states,policy_oracle,value_oracle))

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
	plotter.save_figs("../current/models/dataset_value_l{}.pdf".format(l))
	print('expert demonstration v completed in {}s.'.format(time.time()-start_time))	
	return train_dataset, test_dataset


def datapoints_to_dataset(datapoints,oracle_name,encoding_dim,target_dim,robot=0):
	dataset_fn = get_dataset_fn(oracle_name,l,robot=robot)
	datapoints = np.array(datapoints)
	write_dataset(datapoints,dataset_fn)
	dataset = Dataset(dataset_fn,encoding_dim,target_dim)
	return dataset


def train_model(problem,train_dataset,test_dataset,l,oracle_name,robot=0):
	start_time = time.time()
	print('training model...')

	# device = "cpu"
	device = "cuda"
	model_fn, _ = get_oracle_fn(l,problem.num_robots)

	_, model = get_oracles(problem,value_oracle_name=value_oracle_name,force=True)
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
	plotter.save_figs("../current/models/losses_{}_l{}_i{}.pdf".format(oracle_name,l,robot))
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


# def eval_model(problem,model_fn,l):
	
# 	_, value_oracle = get_oracles(problem,
# 		value_oracle_name = value_oracle_name,
# 		value_oracle_path = model_fn
# 		)

# 	states = []
# 	values = []
# 	for _ in range(num_eval):
# 		state = problem.sample_state()
# 		value = value_oracle.eval(problem,state)
# 		states.append(state)
# 		values.append(value)

# 	states = np.array(states).squeeze(axis=2)
# 	values = np.array(values).squeeze(axis=2)
# 	plotter.plot_value_dataset(problem,[[states,values]],["Eval"])
# 	plotter.save_figs("../current/models/model_eval_l{}.pdf".format(l))

def eval_value(problem,l):
	
	value_oracle_path, policy_oracle_paths = get_oracle_fn(l,problem.num_robots)

	_, value_oracle = get_oracles(problem,
		value_oracle_name = value_oracle_name,
		value_oracle_path = value_oracle_path
		)

	states = []
	values = []
	for _ in range(num_v_eval):
		state = problem.sample_state()
		value = value_oracle.eval(problem,state)
		states.append(state)
		values.append(value)

	states = np.array(states).squeeze(axis=2)
	values = np.array(values).squeeze(axis=2)
	plotter.plot_value_dataset(problem,[[states,values]],["Eval"])
	plotter.save_figs("../current/models/value_eval_l{}.pdf".format(l))


if __name__ == '__main__':

	problem = get_problem(problem_name) 
	format_dir(clean_dirnames=["data","models"]) 

	if batch_size > num_D_v * (1-train_test_split):
		batch_size = int(num_D_v * train_test_split / 5)
		print('changing batch size to {}'.format(batch_size))

	# training 
	for l in range(L):
		start_time = time.time()
		print('learning iteration: {}/{}...'.format(l,L))

		policy_oracle = [None for _ in range(problem.num_robots)]

		if l == 0:
			value_oracle_path = None 
			value_oracle = None
		else: 
			value_oracle_path,_ = get_oracle_fn(l-1,problem.num_robots)
			_,value_oracle = get_oracles(problem,
				value_oracle_name = value_oracle_name,
				value_oracle_path = value_oracle_path
				)

		print('\t value training l/L: {}/{}'.format(l,L))
		train_dataset_v, test_dataset_v = make_expert_demonstration_v(problem, num_D_v, value_oracle, policy_oracle)
		train_model(problem,train_dataset_v,test_dataset_v,l,"value") 
		eval_value(problem,l) 
		print('complete learning iteration: {}/{} in {}s'.format(l,L,time.time()-start_time))

