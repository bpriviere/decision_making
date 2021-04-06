
#pragma once
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>


// Problem settings holds all possible problem parameters 
class Problem_Settings
{
    public: 
        float timestep; 
        float pos_lim; 
        float vel_lim; 
        float gamma; 
        float mass; 
        float acc_lim; 
        float omega_lim;
        float rad_lim; 
        float g; 
        float desired_distance;
        float state_control_weight; 
};


// All internal functions of problem need to be overloaded
class Problem
{
public:
    int m_state_dim;
    int m_action_dim;
    int m_num_robots;
    float m_timestep;
    float m_gamma; 
    virtual ~Problem() { }

    virtual void set_params(Problem_Settings & problem_settings) 
    {
        0; 
    }

    // forward propagate dynamics 
    virtual Eigen::Matrix<float, -1, 1> step(
        Eigen::Matrix<float, -1, 1> state,
        Eigen::Matrix<float, -1, 1> action)
    {
        Eigen::Matrix<float, -1, 1> next_state;
        return next_state;
    }

    // calculate rewards  
    virtual Eigen::Matrix<float, -1, 1> reward(
        Eigen::Matrix<float, -1, 1> state,
        Eigen::Matrix<float, -1, 1> action)
    {
        Eigen::Matrix<float, -1, 1> reward;
        return reward;
    }

    // calculate rewards  
    virtual Eigen::Matrix<float, -1, 1> normalized_reward(
        Eigen::Matrix<float, -1, 1> state,
        Eigen::Matrix<float, -1, 1> action)
    {
        Eigen::Matrix<float, -1, 1> reward;
        return reward;
    }

    // stop condition
    virtual bool is_terminal(Eigen::Matrix<float, -1, 1> state)
    {
        return true;
    }

    // initialize state 
    virtual Eigen::Matrix<float, -1, 1> initialize(std::default_random_engine& generator)
    {
        Eigen::Matrix<float, -1, 1> state;
        return state;
    }

    // action sample  
    virtual Eigen::Matrix<float, -1, 1> sample_action(std::default_random_engine& generator)
    {
        Eigen::Matrix<float, -1, 1> action;
        return action;
    }

    // is valid 
    virtual bool is_valid(Eigen::Matrix<float, -1, 1> state)
    {
        return false; 
    }

};