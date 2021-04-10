
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example2 : public Problem { 
    
    public:
        Eigen::Matrix<float,4,4> m_F;
        Eigen::Matrix<float,4,2> m_B;
        Eigen::Matrix<float,4,4> m_Q;
        Eigen::Matrix<float,2,2> m_R;  
        float m_r_min; 
        float m_r_max;  
        float m_mass; 

        void set_params(Problem_Settings & problem_settings) override 
        {
            m_state_dim = 4;
            m_action_dim = 2;
            m_num_robots = 1;
            m_r_max = 100;
            m_r_min = -1 * m_r_max;  

            problem_settings.state_lims.resize(m_state_dim,2);
            problem_settings.action_lims.resize(m_action_dim,2);
            problem_settings.init_lims.resize(m_state_dim,2);

            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;
            m_state_lims = problem_settings.state_lims;
            m_action_lims = problem_settings.action_lims; 
            m_init_lims = problem_settings.init_lims; 
            m_mass = problem_settings.mass; 

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_F <<  1,0,m_timestep,0,
                    0,1,0,m_timestep,
                    0,0,1,0,
                    0,0,0,1;
            m_B <<  0,0,
                    0,0,
                    m_timestep / m_mass,0,
                    0,m_timestep / m_mass;
            m_Q <<  1,0,0,0,
                    0,1,0,0,
                    0,0,1,0,
                    0,0,0,1;
            m_R <<  1,0,
                    0,1;
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
            next_state = m_F * state + m_B * action; 
            return next_state;
        }


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            r = -1 * (state.transpose() * m_Q * state + action.transpose() * m_R * action); 
            return r;
        }


        Eigen::Matrix<float,-1,1> normalized_reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            r = reward(state,action);
            r = r.cwiseMin(m_r_max).cwiseMax(m_r_min);
            r.array() = (r.array() - m_r_min) / (m_r_max - m_r_min);
            return r;
        }
};
