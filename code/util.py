

import numpy as np 
import importlib
import signal
import json 
import pprint
import pickle 
import os 
import glob 
from tqdm import tqdm 
from queue import Empty 

def dbgp(name,value):
	if type(value) is dict:
		print('{}'.format(name))
		for key_i,value_i in value.items():
			print('{}:{}'.format(str(key_i),value_i))
	else:
		print('{}:{}'.format(name,value))

def load_module(fn):
	module_dir, module_name = fn.split("/")
	module_name, _ = module_name.split(".")
	module = importlib.import_module("{}.{}".format(module_dir, module_name))
	return module	

def write_sim_result(sim_result_dict,fn):
	with open(fn+'.pickle', 'xb') as h:
		pickle.dump(sim_result_dict, h)

def load_sim_result(fn):
	with open(fn, 'rb') as h:
		sim_result = pickle.load(h)
	return sim_result

def write_dataset(dataset,fn):
	# with open(fn, 'xb') as h:
	# 	pickle.dump(dataset, h)
	np.save(fn,dataset)

def get_dataset_fn(oracle_name,l,robot=0):
	# return "../current/data/{}_l{}_i{}.pickle".format(oracle,l,robot)
	return "../current/data/{}_l{}_i{}.npy".format(oracle_name,l,robot)

# def get_oracle_fn(oracle_name,l,robot=0):
	# return "../current/models/model_{}_l{}_i{}.pt".format(oracle_name,l,robot)

def get_oracle_fn(l,num_robots):
	value_oracle_path = "../current/models/model_value_l{}.pt".format(l)
	policy_oracle_paths = []
	for i in range(num_robots):
		policy_oracle_paths.append("../current/models/model_policy_l{}_i{}.pt".format(l,i))
	return value_oracle_path, policy_oracle_paths



def format_dir(clean_dirnames=[]):
	dirnames = ["plots","data","models"]
	for dirname in dirnames:
		path = os.path.join(os.getcwd(),"../current/{}".format(dirname))
		os.makedirs(path,exist_ok=True)
	for dirname in clean_dirnames:
		path = os.path.join(os.getcwd(),"../current/{}".format(dirname))
		for file in glob.glob(path + "/*"):
			os.remove(file)

def sample_vector(lims,damp=0.0):
	# from cube
	dim = lims.shape[0]
	x = np.zeros((dim,1))
	for i in range(dim):
		x[i] = lims[i,0] + np.random.uniform(damp,1-damp)*(lims[i,1] - lims[i,0])
	return x

def contains(vector,lims):
	return (vector[:,0] >= lims[:,0]).all() and (vector[:,0] <= lims[:,1]).all()

def get_temp_fn(dirname,i):
	return "{}/temp_{}.npy".format(dirname,i)


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
