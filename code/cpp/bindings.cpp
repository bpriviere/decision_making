
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include "problems/problem_wrapper.hpp"
#include "solvers/solver_wrapper.hpp"


Solver_Result cpp_search(Problem_Wrapper problem_wrapper, Solver_Wrapper solver_wrapper, Eigen::Matrix<float,-1,1> state, int turn) {
    Solver_Result solver_result; 
    state.resize(problem_wrapper.problem->m_state_dim,1);
    solver_result = solver_wrapper.solver->search(problem_wrapper.problem,state,turn);
    return solver_result;
}


// python interface
PYBIND11_MODULE(bindings, m) {

    m.def("cpp_search", &cpp_search, "PUCT");

    pybind11::class_<Problem_Wrapper> (m, "Problem_Wrapper")
        .def(pybind11::init<std::string,Problem_Settings>());

    pybind11::class_<Problem_Settings> (m, "Problem_Settings")
        .def(pybind11::init())
        .def_readwrite("timestep", &Problem_Settings::timestep)
        .def_readwrite("state_lims", &Problem_Settings::state_lims)
        .def_readwrite("action_lims", &Problem_Settings::action_lims)
        .def_readwrite("init_lims", &Problem_Settings::init_lims)
        .def_readwrite("desired_distance", &Problem_Settings::desired_distance)
        .def_readwrite("r_max", &Problem_Settings::r_max)
        .def_readwrite("g", &Problem_Settings::g)
        .def_readwrite("mass", &Problem_Settings::mass)
        .def_readwrite("state_control_weight", &Problem_Settings::state_control_weight)
        .def_readwrite("gamma", &Problem_Settings::gamma);

    pybind11::class_<Solver_Wrapper> (m, "Solver_Wrapper")
        .def(pybind11::init<std::string,Solver_Settings>());

    pybind11::class_<Solver_Settings> (m, "Solver_Settings")
        .def(pybind11::init())
        .def_readwrite("number_simulations", &Solver_Settings::num_simulations)
        .def_readwrite("search_depth", &Solver_Settings::search_depth)
        .def_readwrite("C_exp", &Solver_Settings::C_exp)
        .def_readwrite("alpha_exp", &Solver_Settings::alpha_exp)
        .def_readwrite("C_pw", &Solver_Settings::C_pw)
        .def_readwrite("alpha_pw", &Solver_Settings::alpha_pw)
        .def_readwrite("beta_policy", &Solver_Settings::beta_policy)
        .def_readwrite("beta_value", &Solver_Settings::beta_value);

    pybind11::class_<Solver_Result> (m, "Solver_Result")
        .def(pybind11::init())
        .def_readwrite("best_action", &Solver_Result::best_action)
        .def_readwrite("child_distribution", &Solver_Result::child_distribution)
        .def_readwrite("tree", &Solver_Result::tree)
        .def_readwrite("value", &Solver_Result::value)
        .def_readwrite("success", &Solver_Result::success);

}
