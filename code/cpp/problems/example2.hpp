
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example2 : public Problem { 
    
    public:
        std::uniform_real_distribution<double> dist;
        std::default_random_engine m_gen;
        Eigen::Matrix<float,4,2> m_state_lims;  
        Eigen::Matrix<float,2,2> m_action_lims;  
        Eigen::Matrix<float,4,4> m_F;
        Eigen::Matrix<float,4,2> m_B;
        Eigen::Matrix<float,4,4> m_Q;
        Eigen::Matrix<float,2,2> m_R;  
        float m_r_min; 
        float m_r_max;  

        void set_params(Problem_Settings & problem_settings) override 
        {
            m_state_dim = 4;
            m_action_dim = 2;
            m_num_robots = 1;
            m_r_max = 100;
            m_r_min = -1 * m_r_max;  

            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;

            float pos_lim = problem_settings.pos_lim;
            float vel_lim = problem_settings.vel_lim;
            float acc_lim = problem_settings.acc_lim; 
            float mass = problem_settings.mass; 

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_state_lims(0,0) = -pos_lim;
            m_state_lims(0,1) = pos_lim;
            m_state_lims(1,0) = -pos_lim;
            m_state_lims(1,1) = pos_lim;
            m_state_lims(2,0) = -vel_lim;
            m_state_lims(2,1) = vel_lim;
            m_state_lims(3,0) = -vel_lim;
            m_state_lims(3,1) = vel_lim;

            m_action_lims(0,0) = -acc_lim;
            m_action_lims(0,1) = acc_lim;
            m_action_lims(1,0) = -acc_lim;
            m_action_lims(1,1) = acc_lim;

            m_F <<  1,0,m_timestep,0,
                    0,1,0,m_timestep,
                    0,0,1,0,
                    0,0,0,1;
            m_B <<  0,0,
                    0,0,
                    m_timestep / mass,0,
                    0,m_timestep / mass;
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
        

        Eigen::Matrix<float,-1,1> sample_state(std::default_random_engine & gen) 
        {
            Eigen::Matrix<float,-1,1> state(m_state_dim,1); 
            for (int ii = 0; ii < m_state_dim; ii++){
                float alpha = dist(gen); 
                state(ii,0) = alpha * (m_state_lims(ii,1) - m_state_lims(ii,0)) + m_state_lims(ii,0);
            }
            return state;
        }


        Eigen::Matrix<float,-1,1> sample_action(std::default_random_engine & gen) override 
        {
            Eigen::Matrix<float,2,1> action(m_action_dim,1); 
            for (int ii = 0; ii < m_action_dim; ii++){
                float alpha = dist(gen); 
                action(ii,0) = alpha * (m_action_lims(ii,1) - m_action_lims(ii,0)) + m_action_lims(ii,0);
            }
            return action;
        } 
        

        Eigen::Matrix<float,-1,1> initialize(std::default_random_engine & gen) override 
        {
            auto state = sample_state(gen); 
            return state;
        }


        bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
        {
            return !is_valid(state);
        }


        bool is_valid(Eigen::Matrix<float,-1,1> state) override 
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }

};
