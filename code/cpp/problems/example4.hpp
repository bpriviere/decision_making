


#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <math.h>       /* sin , cos , tan */
#include "problem.hpp"

class Example4 : public Problem { 
    
    public:
        Eigen::Matrix<float,6,6> m_Q;
        Eigen::Matrix<float,3,3> m_R;
        Eigen::Matrix<float,6,6> m_Fc; 
        Eigen::Matrix<float,6,3> m_Bc; 
        Eigen::Matrix<float,6,6> m_I; 
        float m_r_min; 
        float m_r_max;  
        float m_g; 
        float m_mass; 
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

            m_timestep = problem_settings.timestep;
            m_gamma = problem_settings.gamma;
            m_desired_distance = problem_settings.desired_distance;
            m_mass = problem_settings.mass; 
            m_r_max = problem_settings.r_max;
            m_r_min = problem_settings.r_min;
            m_state_control_weight = problem_settings.state_control_weight;
            m_state_lims = problem_settings.state_lims; 
            m_action_lims = problem_settings.action_lims; 
            m_init_lims = problem_settings.init_lims; 

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_Fc << 0,0,0,1.0f,0,0,
                    0,0,0,0,1.0f,0,
                    0,0,0,0,0,1.0f,
                    0,0,0,0,0,0,
                    0,0,0,0,0,0,
                    0,0,0,0,0,0; 
            
            m_Bc << 0,0,0,
                    0,0,0,
                    0,0,0,
                    1.0f/m_mass,0,0,
                    0,1.0f/m_mass,0,
                    0,0,1.0f/m_mass; 

            m_I.setIdentity();

            // m_Q.setZero(); 
            // m_Q(0,0) = 1;
            // m_Q(1,1) = 1; 
            // m_Q(2,2) = 1; 

            m_Q.setZero();
            // m_Q(0,0) = 1.0 / pow(m_state_lims(0,1) - m_state_lims(0,0), 2.0);
            // m_Q(1,1) = 1.0 / pow(m_state_lims(1,1) - m_state_lims(1,0), 2.0);
            // m_Q(2,2) = 1.0 / pow(m_state_lims(2,1) - m_state_lims(2,0), 2.0);
            // m_Q(3,3) = 1.0 / pow(m_state_lims(3,1) - m_state_lims(3,0), 2.0);
            // m_Q(4,4) = 1.0 / pow(m_state_lims(4,1) - m_state_lims(4,0), 2.0);
            // m_Q(5,5) = 1.0 / pow(m_state_lims(5,1) - m_state_lims(5,0), 2.0);
            // m_Q(0,0) = 1.0 / pow(m_init_lims(0,1) - m_init_lims(0,0), 2.0);
            // m_Q(1,1) = 1.0 / pow(m_init_lims(1,1) - m_init_lims(1,0), 2.0);
            // m_Q(2,2) = 1.0 / pow(m_init_lims(2,1) - m_init_lims(2,0), 2.0);
            // m_Q(0,0) = 1.0 / pow(50.0, 2.0);
            // m_Q(1,1) = 1.0 / pow(50.0, 2.0);
            // m_Q(2,2) = 1.0 / pow(50.0, 2.0);
            m_Q(0,0) = 1.0 / pow(100.0, 2.0);
            m_Q(1,1) = 1.0 / pow(100.0, 2.0);
            m_Q(2,2) = 1.0 / pow(100.0, 2.0);
            // m_Q(3,3) = 1.0 / pow(m_init_lims(3,1) - m_init_lims(3,0), 2.0);
            // m_Q(4,4) = 1.0 / pow(m_init_lims(4,1) - m_init_lims(4,0), 2.0);
            // m_Q(5,5) = 1.0 / pow(m_init_lims(5,1) - m_init_lims(5,0), 2.0);

            m_R.setIdentity();
            m_R = m_R * m_state_control_weight;
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action,
            float timestep) override
        {
            // s = [x,y,z,vx,vy,vz]
            // a = [ax,ay,az]

            Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
            Eigen::Matrix<float,6,6> Fd = m_I + m_Fc * timestep;
            Eigen::Matrix<float,6,3> Bd = m_Bc * timestep; 
            
            // dynamics 
            for (int ii = 0; ii < m_num_robots; ii++){
                // block(i,j,p,q): Block of size (p,q), starting at (i,j) 
                next_state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) = 
                    Fd * state.block(m_state_idxs[ii][0],0,m_state_idxs[ii].size(),1) + 
                    Bd * action.block(m_action_idxs[ii][0],0,m_action_idxs[ii].size(),1);
            }   
            
            // todo: artificial state bound 
            for (int ii = 0; ii < m_state_dim; ii++) {
                next_state(ii,0) = std::max(std::min(next_state(ii,0), m_state_lims(ii,1)), m_state_lims(ii,0));
            }
            // std::cout << "next_state " << next_state << std::endl;

            return next_state;
        }


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            // // old
            // Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            // Eigen::Matrix<float,-1,1> s1 = state.block(m_state_idxs[0][0],0,m_state_idxs[0].size(),1);
            // Eigen::Matrix<float,-1,1> s2 = state.block(m_state_idxs[1][0],0,m_state_idxs[1].size(),1);
            // Eigen::Matrix<float,-1,1> a1 = action.block(m_action_idxs[0][0],0,m_action_idxs[0].size(),1);

            // r(0) = -1 * (abs((s1-s2).transpose() * m_Q * (s1-s2) - m_desired_distance) + a1.transpose() * m_R * a1);
            // r(1) = -1 * r(0); 
            
            // std::cout << "reward fnc" << std::endl;


            // // new
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            Eigen::Matrix<float,-1,1> s1 = state.block(m_state_idxs[0][0],0,m_state_idxs[0].size(),1);
            Eigen::Matrix<float,-1,1> s2 = state.block(m_state_idxs[1][0],0,m_state_idxs[1].size(),1);

            // std::cout << "s1 " << s1 << std::endl;
            // std::cout << "s2 " << s2 << std::endl;
            
            // calc error 
            float x = (s1 - s2).transpose() * m_Q * (s1 - s2); // x in [0,1]

            x = x / 6.0;

            // pursuer gets points if x is small
            r(0,0) = 1.0 - x;
            // evader gets points if x is large
            r(1,0) = x;

            // make sure we are within 0,1 range
            r(0,0) = std::max(std::min(r(0,0), 1.0f), 0.0f);
            r(1,0) = std::max(std::min(r(1,0), 1.0f), 0.0f);


            // // discount purseur if purseur on the boundary 
            // for (int ii = 0; ii < 6; ii++) {
            //     if (s1(ii,0) == m_state_lims(ii,0)) {
            //         r(0,0) = 0.8 * r(0,0);
            //         break;
            //     }
            // }

            // // discount evader if evader on the boundary 
            // for (int ii = 0; ii < 6; ii++) {
            //     if (s2(ii,0) == m_state_lims(ii+6,0)) {
            //         r(1,0) = 0.8 * r(1,0);
            //         break;
            //     }
            // }

            // // discount purseur if evader is outside of heading cone
            // # from: https://www.mathworks.com/matlabcentral/answers/408012-how-to-check-if-a-3d-point-is-inside-a-3d-cone
            // Eigen::Matrix<float,3,1> u = s1.block(3,0,3,0) / s1.norm();
            // Eigen::Matrix<float,3,1> v = s1.block(0,0,3,0);
            // Eigen::Matrix<float,3,1> p = s2.block(0,0,3,0);
            // Eigen::Matrix<float,3,1> vp = (v - p) / (v - p).norm();
            // float angle = std::acos(vp.dot(u));
            // float heading = 35.0f * 3.14f / 180.0f;
            // if (angle > heading) {
            //     r(0,0) = 0.8 * r(0,0);
            // }

            return r;
        }


        Eigen::Matrix<float,-1,1> normalized_reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        {
            // // old
            // Eigen::Matrix<float,-1,1> r = Eigen::Matrix<float,-1,1>::Zero(m_num_robots,1);
            // r = reward(state,action);
            // r = r.cwiseMin(m_r_max).cwiseMax(m_r_min);
            // r.array() = (r.array() - m_r_min) / (m_r_max - m_r_min);
            // r(1) = 1 - r(0);

            // new
            Eigen::Matrix<float,-1,1> r = Eigen::Matrix<float,-1,1>::Zero(m_num_robots,1);
            r = reward(state, action);

            // std::cout << "normalized_reward " << r << std::endl;

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

        Eigen::Matrix<float,-1,1> policy_encoding(Eigen::Matrix<float,-1,1> state, int robot) override 
        {
            return state.block(0,0,6,1) - state.block(6,0,6,1);
        }

        Eigen::Matrix<float,-1,1> value_encoding(Eigen::Matrix<float,-1,1> state) override 
        {
            return state.block(0,0,6,1) - state.block(6,0,6,1);
        }

};
