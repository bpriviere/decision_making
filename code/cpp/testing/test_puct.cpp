
#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../problems/problem.hpp"
#include "../problems/problem_wrapper.hpp"
#include "../solvers/solver.hpp"
#include "../solvers/solver_wrapper.hpp"

int main()
{
    
    // solver settings 
    Solver_Settings solver_settings; 
    solver_settings.num_simulations = 1000;
    solver_settings.search_depth = 10;
    solver_settings.C_exp = 1.0f;
    solver_settings.alpha_exp = 0.25f;
    solver_settings.C_pw = 2.0f;
    solver_settings.alpha_pw = 0.5f;
    solver_settings.beta_policy = 0.0f;
    solver_settings.beta_value = 0.0f; 
    
    // solver wrapper 
    Solver_Wrapper solver_wrapper("C_PUCT_V1",solver_settings);
    
    // problem stuff 
    std::string problem_name = "example1";

    // 
    Problem_Settings problem_settings;
    if (problem_name == "example1") {
        problem_settings.timestep = 0.1;
        problem_settings.gamma = 1.0; 
        problem_settings.state_lims.resize(2,2); 
        problem_settings.state_lims << 
            -5.0,5.0,
            -5.0,5.0;
        problem_settings.action_lims.resize(2,2); 
        problem_settings.action_lims << 
            -1.0,1.0,
            -1.0,1.0;

    } else if (problem_name == "example2") {
        problem_settings.timestep = 0.1; 
        problem_settings.gamma = 1.0; 
        problem_settings.mass = 1.0;
        problem_settings.state_lims.resize(4,2); 
        problem_settings.state_lims << 
            -5.0,5.0,
            -5.0,5.0,
            -1.0,1.0,
            -1.0,1.0;
        problem_settings.action_lims.resize(2,2); 
        problem_settings.action_lims << 
            -1.0,1.0,
            -1.0,1.0;

    } else if (problem_name == "example3") {
        problem_settings.timestep = 0.1;
        problem_settings.gamma = 1.0;
        problem_settings.g = 3;
        problem_settings.desired_distance = 0.2;
        problem_settings.state_lims.resize(14,2); 
        problem_settings.action_lims.resize(6,2); 

        int state_shift;
        int action_shift;
        float pos_lim = 5.0;
        float vel_lim = 1.0; 
        float acc_lim = 0.1; 
        float rad_lim = 2.0*M_PI; 
        float omega_lim = 2.0/10.0*M_PI;
        for (int ii=0; ii<2; ii++){

            state_shift = 7 * ii ; 
            action_shift = 3 * ii ;

            problem_settings.state_lims(state_shift+0,0) = -pos_lim;
            problem_settings.state_lims(state_shift+0,1) = pos_lim;
            problem_settings.state_lims(state_shift+1,0) = -pos_lim;
            problem_settings.state_lims(state_shift+1,1) = pos_lim;
            problem_settings.state_lims(state_shift+2,0) = -pos_lim;
            problem_settings.state_lims(state_shift+2,1) = pos_lim;
            problem_settings.state_lims(state_shift+3,0) = -rad_lim;
            problem_settings.state_lims(state_shift+3,1) = rad_lim;
            problem_settings.state_lims(state_shift+4,0) = -rad_lim;
            problem_settings.state_lims(state_shift+4,1) = rad_lim;
            problem_settings.state_lims(state_shift+5,0) = -rad_lim;
            problem_settings.state_lims(state_shift+5,1) = rad_lim;
            problem_settings.state_lims(state_shift+6,0) = 0.5*vel_lim;
            problem_settings.state_lims(state_shift+6,1) = vel_lim;

            problem_settings.action_lims(action_shift+0,0) = -omega_lim;
            problem_settings.action_lims(action_shift+0,1) = omega_lim;                
            problem_settings.action_lims(action_shift+1,0) = -omega_lim;
            problem_settings.action_lims(action_shift+1,1) = omega_lim;
            problem_settings.action_lims(action_shift+2,0) = -acc_lim;
            problem_settings.action_lims(action_shift+2,1) = acc_lim;
        }
    }
    problem_settings.init_lims = problem_settings.state_lims; 

    Problem_Wrapper problem_wrapper(problem_name,problem_settings);

    auto root_state = problem_wrapper.problem->initialize(solver_wrapper.solver->g_gen); 
    Solver_Result solver_result = solver_wrapper.solver->search(problem_wrapper.problem,root_state,0);

    std::cout << "solver_result.success: " << solver_result.success << std::endl;
    std::cout << "solver_result.best_action: " << solver_result.best_action << std::endl;
    std::cout << "solver_result.child_distribution: " << solver_result.child_distribution << std::endl;

    return 0;
}