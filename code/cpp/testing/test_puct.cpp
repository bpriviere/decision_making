#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../problems/problem.hpp"
#include "../problems/problem_wrapper.hpp"
#include "../solvers/puct.hpp" 

int main()
{
    
    // random generator 
    std::random_device dev;
    std::default_random_engine gen(dev());  

    if (false){
        Example1 problem = Example1(); 
        // problem.set_params();

        // sample
        auto state = problem.initialize(gen);
        auto action = problem.sample_action(gen);

        // validity 
        bool valid = problem.is_valid(state); 

        // function 
        auto next_state = problem.step(state,action);
        auto reward = problem.reward(state,action);
        bool done = problem.is_terminal(state); 

        // print 
        std::cout << "state:" << state << std::endl ;
        std::cout << "action:" << action << std::endl ;
        std::cout << "next_state:" << next_state << std::endl ;
        std::cout << "reward:" << reward << std::endl ;     
        std::cout << "done:" << done << std::endl ;   
    }

    // search settings 
    int num_nodes = 1000; 
    int search_depth = 10;
    float C_exp = 1.0f;
    float alpha_exp = 0.25f; 
    float C_pw = 2.0f; 
    float alpha_pw = 0.5f; 
    float beta_policy = 0.0f; 
    float beta_value = 0.0f; 

    // search
    PUCT puct(gen,num_nodes,search_depth,C_exp,alpha_exp,C_pw,alpha_pw,beta_policy,beta_value);

    // 
    std::string problem_name = "example3";
    Problem_Settings s_p;
    if (problem_name == "example1") {
        s_p.timestep = 0.1; 
        s_p.pos_lim = 5.0;
        s_p.vel_lim = 1.0; 
        s_p.gamma = 1.0; 

    } else if (problem_name == "example2") {
        s_p.timestep = 0.1; 
        s_p.pos_lim = 5.0;
        s_p.vel_lim = 1.0; 
        s_p.gamma = 1.0; 
        s_p.acc_lim = 1.0;
        s_p.mass = 1.0;

    } else if (problem_name == "example3") {
        s_p.timestep = 0.1;
        s_p.pos_lim = 2.0; 
        s_p.vel_lim = 2.0;
        s_p.acc_lim = 1.0;
        s_p.rad_lim = 2*M_PI;
        s_p.omega_lim = 2*M_PI/10.0;
        s_p.gamma = 1.0;
        s_p.g = 3;
        s_p.desired_distance = 0.2;
    }

    Problem_Wrapper w_p = Problem_Wrapper("example3", s_p);

    auto root_state = w_p.problem->initialize(gen); 
    auto root_node = puct.search(w_p.problem,root_state);

    std::cout << "root_node.num_visits: " << root_node.num_visits << std::endl;

    return 0;
}