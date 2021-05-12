

#pragma once
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "../problems/problem.hpp"
#include "../learning/policy_network_wrapper.hpp"
#include "../learning/value_network_wrapper.hpp"

// Solver settings holds all possible solver parameters 
class Solver_Settings
{
    public: 
		int num_simulations;
		int search_depth;
		float C_exp;
		float alpha_exp;
		float C_pw;
		float alpha_pw;
		float beta_policy;
		float beta_value;
        
};

// Solver result holds results 
class Solver_Result
{
    public:
        Eigen::Matrix<float,-1,1> best_action;
        Eigen::Matrix<float,-1,-1> child_distribution;
        Eigen::MatrixXf tree;
        Eigen::Matrix<float,-1,1> value; 
        int num_visits;
        bool success; 
};

// internal functions of solver need to be overloaded
class Solver
{
    public:
		std::default_random_engine g_gen;
        std::vector<Policy_Network_Wrapper> m_policy_network_wrappers;
        Value_Network_Wrapper m_value_network_wrapper;
        virtual ~Solver() { }

        virtual Solver_Result search(Problem * problem, Eigen::Matrix<float,-1,1> root_state, int turn) 
        {
            Solver_Result solver_result;
            return solver_result;
        }

        virtual void set_params(Solver_Settings & solver_settings, 
            std::vector<Policy_Network_Wrapper> & policy_network_wrappers,
            Value_Network_Wrapper & value_network_wrapper)
        {
            0; 
        }
};