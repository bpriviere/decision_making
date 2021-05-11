
#pragma once
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>


// Problem settings holds all possible problem parameters 
class Problem_Settings
{
    public: 
        float timestep; 
        float gamma; 
        float mass; 
        float g; 
        float desired_distance;
        float state_control_weight; 
        float r_max; 
        Eigen::Matrix<float,-1,2> state_lims;
        Eigen::Matrix<float,-1,2> action_lims;
        Eigen::Matrix<float,-1,2> init_lims;
};


// internal functions of problem need to be overloaded
class Problem
{
public:
    int m_state_dim;
    int m_action_dim;
    int m_num_robots;
    float m_timestep;
    float m_gamma; 
    std::uniform_real_distribution<double> dist;
    std::default_random_engine m_gen;    
    Eigen::Matrix<float,-1,2> m_state_lims;  
    Eigen::Matrix<float,-1,2> m_action_lims;  
    Eigen::Matrix<float,-1,2> m_init_lims; 
    virtual ~Problem() { }

    virtual void set_params(Problem_Settings & problem_settings) 
    {
        0; 
    }

    // forward propagate dynamics 
    virtual Eigen::Matrix<float, -1, 1> step(
        Eigen::Matrix<float, -1, 1> state,
        Eigen::Matrix<float, -1, 1> action,
        float timestep)
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


    Eigen::Matrix<float,-1,1> sample_state(std::default_random_engine & gen) 
    {
        Eigen::Matrix<float,-1,1> state(m_state_dim,1); 
        for (int ii = 0; ii < m_state_dim; ii++){
            float alpha = dist(gen); 
            state(ii,0) = alpha * (m_state_lims(ii,1) - m_state_lims(ii,0)) + m_state_lims(ii,0);
        }
        return state;
    }


    Eigen::Matrix<float,-1,1> sample_action(std::default_random_engine & gen) 
    {
        Eigen::Matrix<float,-1,1> action(m_action_dim,1); 
        for (int ii = 0; ii < m_action_dim; ii++){
            float alpha = dist(gen); 
            action(ii,0) = alpha * (m_action_lims(ii,1) - m_action_lims(ii,0)) + m_action_lims(ii,0);
        }
        return action;
    } 
    

    Eigen::Matrix<float,-1,1> initialize(std::default_random_engine & gen) 
    {
        Eigen::Matrix<float,-1,1> state(m_state_dim,1); 
        for (int ii = 0; ii < m_state_dim; ii++){
            float alpha = dist(gen); 
            state(ii,0) = alpha * (m_init_lims(ii,1) - m_init_lims(ii,0)) + m_init_lims(ii,0);
        }
        return state;
    }

    float sample_timestep(std::default_random_engine & gen, float dt) 
    {
        float alpha = dist(gen) - 1.0f; // (-1,0)
        float timestep = dt * pow(10.0f,alpha);
        return timestep; 
    }


    bool is_terminal(Eigen::Matrix<float,-1,1> state)  
    {
        return !is_valid(state);
    }


    bool is_valid(Eigen::Matrix<float,-1,1> state)  
    {
        return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
    }

    Eigen::Matrix<float,-1,1> policy_encoding(Eigen::Matrix<float,-1,1> state, int robot){
        return state;
    }

};