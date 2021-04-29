

#pragma once
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "../problems/problem.hpp"


// Solver settings holds all possible problem parameters 
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
        bool success; 
};

// internal functions of problem need to be overloaded
class Solver
{
    public:
		std::default_random_engine g_gen;
        virtual ~Solver() { }

        virtual Solver_Result search(Problem * problem, Eigen::Matrix<float,-1,1> root_state, int turn) 
        {
            Solver_Result solver_result;
            return solver_result;
        }

        virtual void set_params(Solver_Settings & solver_settings)
        {
            0; 
        }
};