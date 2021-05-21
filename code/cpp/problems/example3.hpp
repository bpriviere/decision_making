


#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <math.h>       /* sin , cos , tan */
#include "problem.hpp"

class Example3 : public Problem { 
    
    public:
        Eigen::Matrix<float,7,7> m_Q;
        Eigen::Matrix<float,3,3> m_R;
        float m_r_min; 
        float m_r_max;  
        float m_g; 
        float m_desired_distance; 
        float m_state_control_weight;

        void set_params(Problem_Settings & problem_settings) override 
        {
            m_state_dim = problem_settings.state_dim;
            m_action_dim = problem_settings.action_dim;
            m_num_robots = problem_settings.num_robots;
            m_state_idxs = problem_settings.state_idxs;
            m_action_idxs = problem_settings.action_idxs;

            // problem_settings.state_lims.resize(m_state_dim,2);
            // problem_settings.action_lims.resize(m_action_dim,2);
            // problem_settings.init_lims.resize(m_state_dim,2);

            m_r_max = problem_settings.r_max;
            m_r_min = problem_settings.r_min;
            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;
            m_g = problem_settings.g;
            m_desired_distance = problem_settings.desired_distance;
            m_state_control_weight = problem_settings.state_control_weight;
            m_state_lims = problem_settings.state_lims; 
            m_action_lims = problem_settings.action_lims; 
            m_init_lims = problem_settings.init_lims; 

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_Q.setZero(); 
            m_Q(0,0) = 1;
            m_Q(1,1) = 1; 
            m_Q(2,2) = 1; 

            m_R.setIdentity();
            m_R = m_R * m_state_control_weight;
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action,
            float timestep) override
        {
            // s = [x,y,z,psi,gamma,phi,v]
            // a = [gammadot, phidot,vdot]

            Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
            
            // dynamics 
            Eigen::Matrix<float,-1,1> state_derv(m_state_dim,1); 
            for (int ii = 0; ii < m_num_robots; ii++){
                state_derv(m_state_idxs[ii][0],0) = state(m_state_idxs[ii][6],0) * cos(state(m_state_idxs[ii][4],0)) * sin(state(m_state_idxs[ii][3],0));
                state_derv(m_state_idxs[ii][1],0) = state(m_state_idxs[ii][6],0) * cos(state(m_state_idxs[ii][4],0)) * cos(state(m_state_idxs[ii][3],0));
                state_derv(m_state_idxs[ii][2],0) = -state(m_state_idxs[ii][6],0) * sin(state(m_state_idxs[ii][4],0));
                state_derv(m_state_idxs[ii][3],0) = m_g / state(m_state_idxs[ii][6],0) * tan(state(m_state_idxs[ii][5],0));
                state_derv(m_state_idxs[ii][4],0) = action(m_action_idxs[ii][0],0);
                state_derv(m_state_idxs[ii][5],0) = action(m_action_idxs[ii][1],0);
                state_derv(m_state_idxs[ii][6],0) = action(m_action_idxs[ii][2],0);
            }   
            next_state = state + state_derv * timestep;

            // wrap angles 
            for (int ii = 0; ii < m_num_robots; ii++){
                next_state(m_state_idxs[ii][3],0) = fmod(next_state(m_state_idxs[ii][3],0), 2*M_PI); 
                next_state(m_state_idxs[ii][4],0) = fmod(next_state(m_state_idxs[ii][4],0), 2*M_PI); 
                next_state(m_state_idxs[ii][5],0) = fmod(next_state(m_state_idxs[ii][5],0), 2*M_PI);
            }
            return next_state;
        }


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            // Eigen::Matrix<float,-1,1> s1 = state(m_state_idxs[0],{0,1});
            // Eigen::Matrix<float,-1,1> s2 = state(m_state_idxs[1],{0,1}); 
            // Eigen::Matrix<float,-1,1> a1 = action(m_action_idxs[0],{0,1});
            Eigen::Matrix<float,-1,1> s1 = state.block(m_state_idxs[0][0],0,m_state_idxs[0].size(),1);
            Eigen::Matrix<float,-1,1> s2 = state.block(m_state_idxs[1][0],0,m_state_idxs[1].size(),1);
            Eigen::Matrix<float,-1,1> a1 = action.block(m_action_idxs[0][0],0,m_action_idxs[0].size(),1);

            r(0) = -1 * (abs((s1-s2).transpose() * m_Q * (s1-s2) - m_desired_distance) + a1.transpose() * m_R * a1);
            r(1) = -1 * r(0); 
            return r;
        }


        Eigen::Matrix<float,-1,1> normalized_reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            Eigen::Matrix<float,-1,1> r = Eigen::Matrix<float,-1,1>::Zero(m_num_robots,1);
            r = reward(state,action);
            r = r.cwiseMin(m_r_max).cwiseMax(m_r_min);
            r.array() = (r.array() - m_r_min) / (m_r_max - m_r_min);
            r(1) = 1 - r(0);
            return r;
        }


        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }

        bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
        {
            return !is_valid(state);
        }

};
