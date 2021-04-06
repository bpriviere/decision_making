


#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <math.h>       /* sin , cos , tan */
#include "problem.hpp"

class Example3 : public Problem { 
    
    public:
        std::uniform_real_distribution<double> dist;
        std::default_random_engine m_gen;
        Eigen::Matrix<float,14,2> m_state_lims;  
        Eigen::Matrix<float,6,2> m_action_lims;  
        Eigen::Matrix<float,7,7> m_Q;
        Eigen::Matrix<float,3,3> m_R;
        float m_r_min; 
        float m_r_max;  
        int m_state_dim_per_robot;
        int m_action_dim_per_robot;
        float m_g; 
        float m_desired_distance; 
        float m_state_control_weight;

        void set_params(Problem_Settings & problem_settings) override 
        {
            m_state_dim = 14;
            m_state_dim_per_robot = 7; 
            m_action_dim = 6;
            m_action_dim_per_robot = 3; 
            m_num_robots = 2;
            m_r_max = 100;
            m_r_min = -1 * m_r_max;  

            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;
            m_g = problem_settings.g;
            m_desired_distance = problem_settings.desired_distance;
            m_state_control_weight = problem_settings.state_control_weight;

            float pos_lim = problem_settings.pos_lim;
            float vel_lim = problem_settings.vel_lim;
            float acc_lim = problem_settings.acc_lim; 
            float rad_lim = problem_settings.rad_lim;
            float omega_lim = problem_settings.omega_lim;

            std::uniform_real_distribution<double> dist(0,1.0f); 

            int state_shift;
            int action_shift;
            for (int ii=0; ii<m_num_robots; ii++){

                state_shift = m_state_dim_per_robot * ii ; 
                action_shift = m_action_dim_per_robot * ii ;

                m_state_lims(state_shift+0,0) = -pos_lim;
                m_state_lims(state_shift+0,1) = pos_lim;
                m_state_lims(state_shift+1,0) = -pos_lim;
                m_state_lims(state_shift+1,1) = pos_lim;
                m_state_lims(state_shift+2,0) = -pos_lim;
                m_state_lims(state_shift+2,1) = pos_lim;
                m_state_lims(state_shift+3,0) = -rad_lim;
                m_state_lims(state_shift+3,1) = rad_lim;
                m_state_lims(state_shift+4,0) = -rad_lim;
                m_state_lims(state_shift+4,1) = rad_lim;
                m_state_lims(state_shift+5,0) = -rad_lim;
                m_state_lims(state_shift+5,1) = rad_lim;
                m_state_lims(state_shift+6,0) = 0.5*vel_lim;
                m_state_lims(state_shift+6,1) = vel_lim;

                m_action_lims(action_shift+0,0) = -omega_lim;
                m_action_lims(action_shift+0,1) = omega_lim;                
                m_action_lims(action_shift+1,0) = -omega_lim;
                m_action_lims(action_shift+1,1) = omega_lim;
                m_action_lims(action_shift+2,0) = -acc_lim;
                m_action_lims(action_shift+2,1) = acc_lim;
            }

            m_Q.setZero(); 
            m_Q(0,0) = 1;
            m_Q(1,1) = 1; 
            m_Q(2,2) = 1; 

            m_R.setIdentity();
            m_R = m_R * m_state_control_weight;
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            // s = [x,y,z,psi,gamma,phi,v]
            // a = [gammadot, phidot,vdot]

            Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
            
            // dynamics 
            Eigen::Matrix<float,-1,1> state_derv(m_state_dim,1); 
            int state_shift;
            int action_shift;
            for (int ii = 0; ii < m_num_robots; ii++){
                state_shift = ii * m_state_dim_per_robot;
                action_shift = ii * m_action_dim_per_robot;
                state_derv(state_shift+0,0) = state(state_shift+6,0) * cos(state(state_shift+4,0)) * sin(state(state_shift+3,0));
                state_derv(state_shift+1,0) = state(state_shift+6,0) * cos(state(state_shift+4,0)) * cos(state(state_shift+3,0));
                state_derv(state_shift+2,0) = -state(state_shift+6,0) * sin(state(state_shift+4,0));
                state_derv(state_shift+3,0) = m_g / state(state_shift+6,0) * tan(state(state_shift+5,0));
                state_derv(state_shift+4,0) = action(action_shift+0,0);
                state_derv(state_shift+5,0) = action(action_shift+1,0);
                state_derv(state_shift+6,0) = action(action_shift+2,0);
            }   
            next_state = state + state_derv * m_timestep;

            // wrap angles 
            for (int ii = 0; ii < m_num_robots; ii++){
                state_shift = ii * m_state_dim_per_robot;
                next_state(state_shift+3,0) = fmod(next_state(state_shift+3,0), 2*M_PI); 
                next_state(state_shift+4,0) = fmod(next_state(state_shift+4,0), 2*M_PI); 
                next_state(state_shift+5,0) = fmod(next_state(state_shift+5,0), 2*M_PI);
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
