
#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "../problems/example1.hpp"
#include "../solvers/puct.hpp" 
 
int main()
{
    Example1 problem; 

    // sample
    auto state = problem.sample_state();
    auto action = problem.sample_action();

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
    PUCT puct(problem,num_nodes,search_depth,C_exp,alpha_exp,C_pw,alpha_pw,beta_policy,beta_value);
    auto result = puct.search(state); 
    auto tree = puct.export_tree();

    std::cout << "result:" << result.total_value/result.num_visits << std::endl ;

    return 0;
}