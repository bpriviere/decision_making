

import numpy as np 
import scipy.io 
import sys 
sys.path.append("../")

from param import Param 
from run import make_instance, run_instance

def relative_pos_to_state(pos_i):
	# assume s2 is at (0,0,0), pos_i = s_2-s_1
	state = np.zeros((12,1))
	state[0,0] = -pos_i[0]
	state[1,0] = -pos_i[1]
	state[2,0] = -pos_i[2]
	return state * 0.1


if __name__ == '__main__':

	# prepare data 
	path_to_6dof_data = '../../saved/EstimatedValues6Hz.mat'
	data = scipy.io.loadmat(path_to_6dof_data)
	pos = data["EstimatedPos"]

	# prepare problem/solver 
	param = Param() 
	param.problem_name = "example4" # only example4 is supported 
	param.solver_name = "C_PUCT"
	instance = make_instance(param)
	instance["problem"].tf = 5 
	instance["problem"].times = np.arange(
		instance["problem"].t0,
		instance["problem"].tf,
		instance["problem"].dt)
	instance["initial_state"] = relative_pos_to_state(pos[0,:])

	sim_result = run_instance(instance)
	print('sim_result["states"][:,0:3]',sim_result["states"][:,0:3])

	

