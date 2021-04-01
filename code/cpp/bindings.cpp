
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "problems/problem_wrapper.hpp"
#include "solvers/puct.hpp"

class PUCT_Wrapper
{
    public: 
        PUCT m_puct; 
        std::default_random_engine m_gen;
        bool m_vis_on;

    PUCT_Wrapper(
        int num_nodes,
        int search_depth,
        float C_exp,
        float alpha_exp,
        float C_pw,
        float alpha_pw,
        float beta_policy,
        float beta_value,
        bool vis_on)
        : m_puct(m_gen
                ,num_nodes
                ,search_depth
                ,C_exp
                ,alpha_exp
                ,C_pw
                ,alpha_pw
                ,beta_policy
                ,beta_value)
        , m_vis_on(vis_on)
        {
            std::random_device dev;
            std::default_random_engine m_gen(dev());
        }
};

class Result
{
public:
    Eigen::Matrix<float,2,1> best_action;
    Eigen::MatrixXf tree;
};

Result search(
    PUCT_Wrapper & w_puct,
    Problem_Wrapper & w_problem,
    Eigen::Matrix<float,2,1> & state)
    {
        Result result;

        // check valid 
        if (!w_problem.problem->is_valid(state)) {
            std::cout << "initial state not valid" << std::endl;
            return result; 
        }

        // search
        auto root_node = w_puct.m_puct.search(w_problem.problem,state); 

        // result 
        result.best_action = w_puct.m_puct.most_visited(&root_node,0)->action_to_node;
        if (w_puct.m_vis_on) {
            result.tree = w_puct.m_puct.export_tree(w_problem.problem); 
        }
        return result;
}

// python interface
PYBIND11_MODULE(bindings, m) {

    m.def("search", &search, "PUCT");

    pybind11::class_<Result> (m, "Result")
        .def(pybind11::init())
        .def_readwrite("best_action", &Result::best_action)
        .def_readwrite("tree", &Result::tree);

    pybind11::class_<PUCT_Wrapper> (m, "PUCT_Wrapper")
        .def(pybind11::init<int, int, float, float, float, float, float, float, bool>());

    pybind11::class_<Problem_Wrapper> (m, "Problem_Wrapper")
        .def(pybind11::init<std::string,Problem_Settings>());

    pybind11::class_<Problem_Settings> (m, "Problem_Settings")
        .def(pybind11::init())
        .def_readwrite("timestep", &Problem_Settings::timestep)
        .def_readwrite("pos_lim", &Problem_Settings::pos_lim)
        .def_readwrite("vel_lim", &Problem_Settings::vel_lim)
        .def_readwrite("gamma", &Problem_Settings::gamma);
}
