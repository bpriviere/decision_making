


#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <math.h>       /* sin , cos , tan */
#include "problem.hpp"

class Example4 : public Problem { 
    
    public:
        std::uniform_real_distribution<double> dist;
        std::default_random_engine m_gen;
        Eigen::Matrix<float,12,2> m_state_lims;  
        Eigen::Matrix<float,6,2> m_action_lims;  
        Eigen::Matrix<float,12,2> m_init_lims;  
        Eigen::Matrix<float,6,6> m_Q;
        Eigen::Matrix<float,3,3> m_R;
        Eigen::Matrix<float,6,6> m_F; 
        Eigen::Matrix<float,6,3> m_B; 
        float m_r_min; 
        float m_r_max;  
        int m_state_dim_per_robot;
        int m_action_dim_per_robot;
        float m_g; 
        float m_mass; 
        float m_desired_distance; 
        float m_state_control_weight;

        void set_params(Problem_Settings & problem_settings) override 
        {
            m_state_dim = 12;
            m_state_dim_per_robot = 6; 
            m_action_dim = 6;
            m_action_dim_per_robot = 3; 
            m_num_robots = 2;
            m_r_max = 100;
            m_r_min = -1 * m_r_max;  

            problem_settings.state_lims.resize(m_state_dim,2);
            problem_settings.action_lims.resize(m_action_dim,2);
            problem_settings.init_lims.resize(m_state_dim,2);

            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;
            m_mass = problem_settings.mass; 
            m_desired_distance = problem_settings.desired_distance;
            m_state_lims = problem_settings.state_lims; 
            m_action_lims = problem_settings.action_lims; 
            m_init_lims = problem_settings.init_lims; 

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_F <<  1,0,0,m_timestep,0,0,
                    0,1,0,0,m_timestep,0,
                    0,0,1,0,0,m_timestep,
                    0,0,0,1,0,0,
                    0,0,0,0,1,0,
                    0,0,0,0,0,1; 
            
            m_B <<  0,0,0,
                    0,0,0,
                    0,0,0,
                    m_timestep/m_mass,0,0,
                    0,m_timestep/m_mass,0,
                    0,0,m_timestep/m_mass; 

            m_Q.setZero(); 
            m_Q(0,0) = 1;
            m_Q(1,1) = 1; 
            m_Q(2,2) = 1; 

            m_R.setIdentity();
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            // s = [x,y,z,vx,vy,vz]
            // a = [ax,ay,az]

            Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
            
            // dynamics 
            int state_shift;
            int action_shift;
            for (int ii = 0; ii < m_num_robots; ii++){
                state_shift = ii * m_state_dim_per_robot;
                action_shift = ii * m_action_dim_per_robot;
                next_state.block(state_shift,0,m_state_dim_per_robot,1) = 
                    m_F * state.block(state_shift,0,m_state_dim_per_robot,1) + m_B * action.block(action_shift,0,m_action_dim_per_robot,1);
            }   
            return next_state;
        }


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            Eigen::Matrix<float,-1,1> s1 = state.head(m_state_dim_per_robot);
            Eigen::Matrix<float,-1,1> s2 = state.tail(m_state_dim_per_robot);
            Eigen::Matrix<float,-1,1> a1 = action.head(m_action_dim_per_robot);

            r(0) = -1 * (abs((s1-s2).transpose() * m_Q * (s1-s2) - m_desired_distance) + a1.transpose() * m_R * a1);
            r(1) = -1 * r(0); 
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
            r(1) = 1 - r(0);
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
            Eigen::Matrix<float,-1,1> action(m_action_dim,1); 
            for (int ii = 0; ii < m_action_dim; ii++){
                float alpha = dist(gen); 
                action(ii,0) = alpha * (m_action_lims(ii,1) - m_action_lims(ii,0)) + m_action_lims(ii,0);
            }
            return action;
        } 
        

        Eigen::Matrix<float,-1,1> initialize(std::default_random_engine & gen) override 
        {
            Eigen::Matrix<float,-1,1> state(m_state_dim,1); 
            for (int ii = 0; ii < m_state_dim; ii++){
                float alpha = dist(gen); 
                state(ii,0) = alpha * (m_init_lims(ii,1) - m_init_lims(ii,0)) + m_init_lims(ii,0);
            }
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
