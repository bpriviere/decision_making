
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "problems/example1.hpp"
#include "solvers/puct.hpp"

// cpp code

// helper 
class Settings
{
public:
  Settings()
    : num_nodes(1000)
    , search_depth(10)
    , C_exp(1.0f)
    , alpha_exp(0.25f)
    , C_pw(2.0f)
    , alpha_pw(0.5f)
    , beta_policy(0.0)
    , beta_value(0.0)
    , vis_on(false)
  {
  }
    int num_nodes;
    int search_depth;
    float C_exp;
    float alpha_exp;
    float C_pw;
    float alpha_pw;
    float beta_policy;
    float beta_value;
    bool vis_on; 
};

class Result
{
public:
    Eigen::Matrix<float,2,1> best_action;
    Eigen::MatrixXf tree;
};

// primary function 
Result search(
    const Eigen::Matrix<float,2,1> &state,
    const Settings &settings){

    Example1 problem; 
    Result result;

    // check valid 
    if (!problem.is_valid(state)) {
        std::cout << "initial state not valid" << std::endl;
        return result; 
    }

    // search settings 
    int num_nodes = settings.num_nodes;
    int search_depth = settings.search_depth;
    float C_exp = settings.C_exp;
    float alpha_exp = settings.alpha_exp;
    float C_pw = settings.C_pw;
    float alpha_pw = settings.alpha_pw;
    float beta_policy = settings.beta_policy;
    float beta_value = settings.beta_value;

    // search
    PUCT puct(problem,num_nodes,search_depth,C_exp,alpha_exp,C_pw,alpha_pw,beta_policy,beta_value);
    auto root_node = puct.search(state); 

    // result 
    result.best_action = puct.most_visited(&root_node,0)->action_to_node;
    if (settings.vis_on) {
        result.tree = puct.export_tree(); 
    }
    
    return result;
}

// python interface
PYBIND11_MODULE(bindings, m) {

    m.def("cpp_search", &search, "PUCT");

    pybind11::class_<Settings> (m, "Settings")
        .def(pybind11::init())
        .def_readwrite("num_nodes", &Settings::num_nodes)
        .def_readwrite("search_depth", &Settings::search_depth)
        .def_readwrite("C_exp", &Settings::C_exp)
        .def_readwrite("alpha_exp", &Settings::alpha_exp)
        .def_readwrite("C_pw", &Settings::C_pw)
        .def_readwrite("alpha_pw", &Settings::alpha_pw)
        .def_readwrite("beta_policy", &Settings::beta_policy)
        .def_readwrite("beta_value", &Settings::beta_value)
        .def_readwrite("vis_on", &Settings::vis_on);

    pybind11::class_<Result> (m, "Result")
        .def(pybind11::init())
        .def_readwrite("best_action", &Result::best_action)
        .def_readwrite("tree", &Result::tree);
}