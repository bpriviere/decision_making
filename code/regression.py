
import time 
import sys 
import numpy as np 
sys.path.append("../")

from run import run_instance, make_instance
from param import Param 
import plotter 


def get_unique_key(param,trial=None):
    if trial is None: 
        trial = param.trial
    return "{}, {}, {}, {}".format(
        param.problem_name,
        param.solver_name,
        param.number_simulations,
        trial)


if __name__ == '__main__':

    mode = 0 

    # fast 
    if mode == 0:
        number_simulations_lst = [100]
        problem_name_lst = ["example1","example2","example4"]
        solver_name_lst = ["PUCT","C_PUCT"]
        num_trial = 1 
    
    # speed test 
    elif mode == 1:
        number_simulations_lst = [10,100,1000,10000]
        problem_name_lst = ["example1"]
        solver_name_lst = ["PUCT","C_PUCT"]
        num_trial = 1 

    params = []
    instances = [] 
    for problem_name in problem_name_lst: 
        for trial in range(num_trial):
            for i_ns, number_simulations in enumerate(number_simulations_lst): 
                for i_sn, solver_name in enumerate(solver_name_lst):
                    param = Param()
                    param.number_simulations = number_simulations
                    param.problem_name = problem_name 
                    param.solver_name = solver_name 
                    param.trial = trial
                    param.number_simulations_lst = number_simulations_lst
                    param.problem_name_lst = problem_name_lst
                    param.solver_name_lst = solver_name_lst
                    param.key = get_unique_key(param)
                    params.append(param)

                    instance = make_instance(param)
                    if i_ns == 0 and i_sn == 0:
                        initial_state = instance["initial_state"]
                    else: 
                        instance["initial_state"] = initial_state
                    instances.append(instance)

    sim_results = []  
    for i,(param,instance) in enumerate(zip(params,instances)): 
        print('running {}/{}'.format(i,len(params)))
        start = time.time()
        sim_result = run_instance(instance,verbose=False)
        duration = time.time() - start
        sim_result["duration_per_timestep"] = duration / len(sim_result["times"])
        sim_results.append((param,sim_result))

    plotter.plot_regression_test(sim_results)
    plotter.save_figs("../current/plots/regression.pdf")
    plotter.open_figs("../current/plots/regression.pdf")
