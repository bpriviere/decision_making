
#include <iostream>
#include <string>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "problems/example1.hpp"
#include "solvers/puct.hpp" 
 
int main()
{
    Example1 m_problem; 

    // sample
    auto state = m_problem.sample_state();
    auto action = m_problem.sample_action();

    // function 
    auto next_state = m_problem.step(state,action);
    auto reward = m_problem.reward(state,action);
    bool done = m_problem.is_terminal(state); 

    // search 
    PUCT m_puct;
    auto result = m_puct.search(state); 

    // print 
    std::cout << "state:" << state << std::endl ;
    std::cout << "action:" << action << std::endl ;
    std::cout << "next_state:" << next_state << std::endl ;
    std::cout << "reward:" << reward << std::endl ;     
    std::cout << "done:" << done << std::endl ;     
    std::cout << "result:" << result.total_value/result.num_visits << std::endl ;

    return 0;
}