
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "problems/example1.hpp"
#include "solvers/puct.hpp"


namespace py = pybind11;

Eigen::Matrix<float,2,1> search(const Eigen::Matrix<float,2,1> &state){

    Example1 problem; 

    // check valid 
    if (!problem.is_valid(state)) {
        std::cout << "initial state not valid" << std::endl;
        return problem.sample_action(); 
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
    PUCT puct(problem,num_nodes,search_depth,C_exp,alpha_exp,C_pw,alpha_pw,beta_policy,beta_value);
    auto result = puct.search(state); 
    auto most_visited_child = puct.most_visited(&result,0);

    return most_visited_child->action_to_node; 
}

PYBIND11_MODULE(bindings, m) {
    m.def("cpp_search", &search, "PUCT");
}