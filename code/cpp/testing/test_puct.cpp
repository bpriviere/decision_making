
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

    // problem stuff 
    std::string problem_name = "example8";
    std::string solver_name = "C_PUCT_V1";

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
    } else if (problem_name == "example6"){
        problem_settings.state_lims.resize(2,2);
        problem_settings.state_lims << 
            -2.0,2.0,
            -2.0,2.0;
        problem_settings.action_lims.resize(2,2);
        problem_settings.action_lims << 
            -0.5,0.5,
            -0.5,0.5;
        problem_settings.init_lims.resize(2,2);
        problem_settings.init_lims << 
            -2.0,2.0,
            -2.0,2.0;
        // problem_settings.timestep = 0.1f;
        problem_settings.timestep = 1.0f;
        problem_settings.gamma = 0.99f;
        // problem_settings.r_max = 10;
        problem_settings.r_max = 1;
        problem_settings.state_control_weight = 1.0f;
        problem_settings.desired_distance = 0.2;
        
        Eigen::Matrix<float,2,2> obstacle1;
        Eigen::Matrix<float,2,2> obstacle2;
        Eigen::Matrix<float,2,2> obstacle3;
        obstacle1 <<
             1.0, 2.0,
            -2.0,-2.0;
        obstacle2 << 
            -2.0, 2.0,
             1.0, 2.0;
        obstacle3 << 
            -2.0, 2.0,
            -2.0,-1.0;

        problem_settings.obstacles.resize(3);
        problem_settings.obstacles[0] = obstacle1;
        problem_settings.obstacles[1] = obstacle2;
        problem_settings.obstacles[2] = obstacle3;

    } else if (problem_name == "example8"){
        problem_settings.state_lims.resize(5,2);
        problem_settings.state_lims << 
            -2.0,2.0,
            -2.0,2.0,
            -2.0,2.0,
            -2.0,2.0,
             0.0, 20.0;
        problem_settings.action_lims.resize(4,2);
        problem_settings.action_lims << 
            -0.5,0.5,
            -0.5,0.5,
            -0.5,0.5,
            -0.5,0.5;
        problem_settings.init_lims.resize(5,2);
        problem_settings.init_lims << 
            -2.0,2.0,
            -2.0,2.0,
            -2.0,2.0,
            -2.0,2.0,
             0.0,0.0;

        problem_settings.state_idxs = {{0,1},{2,3}};
        problem_settings.action_idxs = {{0,1},{2,3}};
        problem_settings.state_dim = 4;
        problem_settings.action_dim = 4;
        problem_settings.num_robots = 2;

        problem_settings.timestep = 0.1f;
        problem_settings.gamma = 0.99f;
        problem_settings.r_max = 1.0f;
        problem_settings.state_control_weight = 1.0f;
        problem_settings.desired_distance = 0.2f;

    }
    problem_settings.init_lims = problem_settings.state_lims; 
    problem_settings.tf = 20.0f;
    problem_settings.r_min = -1 * problem_settings.r_max;
    Problem_Wrapper problem_wrapper(problem_name,problem_settings);
    
    // solver settings 
    Solver_Settings solver_settings; 
    solver_settings.num_simulations = 500;
    solver_settings.search_depth = 10;
    solver_settings.C_exp = 1.0f;
    solver_settings.alpha_exp = 0.25f;
    solver_settings.C_pw = 2.0f;
    solver_settings.alpha_pw = 0.5f;
    solver_settings.beta_policy = 0.5f;
    solver_settings.beta_value = 0.0f; 
    
    // oracles 
    bool oracles_on = false;
    // policy oracles 
    std::vector<Policy_Network_Wrapper> policy_network_wrappers(problem_wrapper.problem->m_num_robots);
    if (oracles_on){
        Eigen::Matrix<float,-1,-1> policy_weight(problem_wrapper.problem->m_state_dim,problem_wrapper.problem->m_action_dim);
        Eigen::Matrix<float,-1, 1> policy_bias(problem_wrapper.problem->m_action_dim); 
        policy_weight.setZero();
        policy_bias.setZero();
        policy_network_wrappers[0].initialize("gaussian");
        policy_network_wrappers[0].addLayer(policy_weight,policy_bias);
    }

    // value oracles
    Value_Network_Wrapper value_network_wrapper;
    if (oracles_on){
        Eigen::Matrix<float,-1,-1> value_weight(problem_wrapper.problem->m_state_dim,problem_wrapper.problem->m_num_robots);
        Eigen::Matrix<float,-1, 1> value_bias(problem_wrapper.problem->m_num_robots); 
        value_weight.setZero();
        value_bias.setZero();
        value_network_wrapper.initialize("deterministic");
        value_network_wrapper.addLayer(value_weight,value_bias);
    }

    // solver wrapper 
    Solver_Wrapper solver_wrapper(solver_name,solver_settings,policy_network_wrappers,value_network_wrapper);
    

    auto root_state = problem_wrapper.problem->initialize(solver_wrapper.solver->g_gen); 
    Solver_Result solver_result = solver_wrapper.solver->search(problem_wrapper.problem,root_state,0);

    std::cout << "solver_result.success: " << solver_result.success << std::endl;
    std::cout << "solver_result.num_visits: " << solver_result.num_visits << std::endl;
    // std::cout << "solver_result.best_action: " << solver_result.best_action << std::endl;
    // std::cout << "solver_result.child_distribution: " << solver_result.child_distribution << std::endl;

    return 0;
}