

import numpy as np 
import importlib
import signal
import json 
import pprint

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