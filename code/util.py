

import numpy as np 
import importlib
import signal
import json 
import pprint
import pickle 
import os 
import glob 

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

def write_dataset(dataset,fn):
	# with open(fn, 'xb') as h:
	# 	pickle.dump(dataset, h)
	np.save(fn,dataset)

def get_dataset_fn(oracle,l,robot=0):
	# return "../current/data/{}_l{}_i{}.pickle".format(oracle,l,robot)
	return "../current/data/{}_l{}_i{}.npy".format(oracle,l,robot)

def get_oracle_fn(oracle,l,robot=0):
	return "../current/models/{}_l{}_i{}.pt".format(oracle,l,robot)

def format_dir(clean_dirnames=[]):
	dirnames = ["plots","data","models"]
	for dirname in dirnames:
		path = os.path.join(os.getcwd(),"../current/{}".format(dirname))
		os.makedirs(path,exist_ok=True)
	for dirname in clean_dirnames:
		path = os.path.join(os.getcwd(),"../current/{}".format(dirname))
		for file in glob.glob(path + "/*"):
			os.remove(file)

