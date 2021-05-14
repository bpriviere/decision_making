
import time 
import sys 
import numpy as np 
sys.path.append("../")

from param import Param 
from run import run_instance, make_instance
import plotter 


def get_unique_key(param,trial=None):
    if trial is None: 
        trial = param.trial
    return "{}, {}, {}, {}".format(
        param.problem_name,
        param.solver_name,
        param.number_simulations,
        trial)

def worker(param):
    from run import run_instance, make_instance
    print('running {}/{}: {}'.format(param.count,param.total,param.key))
    instance = make_instance(param)
    instance["initial_state"] = param.initial_state
    start = time.time()
    
    sim_result = run_instance(instance,verbose=False)
    duration = time.time() - start
    sim_result["duration_per_timestep"] = duration / len(sim_result["times"])
    # remove solver from instance because it can't pickle python bindings 
    del sim_result["instance"]["solver"] 
    return (param,sim_result)


if __name__ == '__main__':

    mode = 3

    # complete
    if mode == 0:
        number_simulations_lst = [50,100,200]
        # problem_name_lst = ["example1","example2",["example4"]
        problem_name_lst = ["example4"]
        # solver_name_lst = ["PUCT","C_PUCT"]
        # solver_name_lst = ["PUCT_V0","C_PUCT_V0","PUCT_V1","C_PUCT_V1"]
        solver_name_lst = ["PUCT_V1","C_PUCT_V1"] #,"PUCT_V1","C_PUCT_V1"]
        num_trial = 20
    
    # speed test 
    elif mode == 1:
        number_simulations_lst = [50,100] #,500,1000] 
        problem_name_lst = ["example1","example2"] #,"example4"] #,"example2","example4"]
        solver_name_lst = ["PUCT_V1","PUCT_V2","C_PUCT_V1"]
        num_trial = 5

    # custom 
    elif mode == 2:
        number_simulations_lst = [100]
        problem_name_lst = ["example6"] #,"example5"]
        # solver_name_lst = ["PUCT","PUCT_PW_MAB"]
        solver_name_lst = ["PUCT_V1"]
        num_trial = 20

    # test for bugs  
    elif mode == 3:
        # problem_name_lst = ["example1","example2","example3","example4","example6","example8"]
        problem_name_lst = ["example1","example2","example3","example4","example5","example6","example8"]
        number_simulations_lst = [1]
        solver_name_lst = ["PUCT_V1","C_PUCT_V1"]
        num_trial = 1
        # solver_name_lst = ["PUCT_V0","PUCT_V1","C_PUCT_V0","C_PUCT_V1"]


    params = []
    total = len(problem_name_lst) * len(number_simulations_lst) * len(solver_name_lst) * num_trial
    count = 0 
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
                    param.num_trial = num_trial
                    param.total = total
                    param.count = count  
                    param.key = get_unique_key(param)

                    instance = make_instance(param)
                    if i_ns == 0 and i_sn == 0:
                        initial_state = instance["initial_state"]
                    else: 
                        instance["initial_state"] = initial_state

                    param.initial_state = initial_state
                    params.append(param)
                    count += 1 

    parallel_on = True

    if parallel_on:
        import multiprocessing as mp
        pool = mp.Pool(mp.cpu_count()-1)
        sim_results = pool.map(worker, params) 

    else:
        sim_results = []  
        for i,(param,instance) in enumerate(params):
            sim_results.append(worker(param)) 

    plotter.plot_regression_test(sim_results,render_on=True)
    plotter.save_figs("../../current/plots/regression.pdf")
    plotter.open_figs("../../current/plots/regression.pdf")
