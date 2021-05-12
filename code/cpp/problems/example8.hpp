
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example8 : public Problem { 
	
	public:
		Eigen::Matrix<float,2,2> m_Fc;
		Eigen::Matrix<float,2,2> m_Bc;
		Eigen::Matrix<float,2,2> m_I; 
		Eigen::Matrix<float,2,2> m_Q;
		Eigen::Matrix<float,2,2> m_R;  
		float m_r_max; 
		float m_r_min; 
		float m_state_control_weight;
		float m_dist;
		int m_state_dim_per_robot; 
		int m_action_dim_per_robot;

		void set_params(Problem_Settings & problem_settings) override 
		{
			m_state_dim_per_robot = 2;
			m_action_dim_per_robot = 2;
			m_num_robots = 2;
			m_state_dim = m_state_dim_per_robot * m_num_robots;
			m_action_dim = m_action_dim_per_robot * m_num_robots;

			problem_settings.state_lims.resize(m_state_dim,2);
			problem_settings.action_lims.resize(m_action_dim,2);
			problem_settings.init_lims.resize(m_state_dim,2);

			m_timestep = problem_settings.timestep;
			m_gamma = problem_settings.gamma;
			m_r_max = problem_settings.r_max;
			m_r_min = -1 * m_r_max;  
			m_state_control_weight = problem_settings.state_control_weight;
			m_state_lims = problem_settings.state_lims; 
			m_action_lims = problem_settings.action_lims; 
			m_init_lims = problem_settings.init_lims;
			m_dist = problem_settings.desired_distance;

			std::uniform_real_distribution<double> dist(0,1.0f); 

			m_Fc.setZero();
			m_Bc.setIdentity();
			m_I.setIdentity();

			m_Q.setIdentity();
			m_R.setIdentity();
			m_R = m_R * m_state_control_weight;
		}


		Eigen::Matrix<float,-1,1> step(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			float timestep) override
		{
			Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
			Eigen::Matrix<float,2,2> Fd = m_I + m_Fc * timestep;
			Eigen::Matrix<float,2,2> Bd = m_Bc * timestep; 

            // dynamics 
            int state_shift;
            int action_shift;
            for (int ii = 0; ii < m_num_robots; ii++){
                state_shift = ii * m_state_dim_per_robot;
                action_shift = ii * m_action_dim_per_robot;
                // block(i,j,p,q): Block of size (p,q), starting at (i,j) 
                next_state.block(state_shift,0,m_state_dim_per_robot,1) = 
                    Fd * state.block(state_shift,0,m_state_dim_per_robot,1) + 
                    Bd * action.block(action_shift,0,m_action_dim_per_robot,1);
            }   
            return next_state;
		}


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
        { 
            Eigen::Matrix<float,-1,1> r(m_num_robots,1);
            Eigen::Matrix<float,-1,1> s1 = state.block(0,0,m_state_dim_per_robot,1);
            Eigen::Matrix<float,-1,1> s2 = state.block(m_state_dim_per_robot,0,m_state_dim_per_robot,1);
            Eigen::Matrix<float,-1,1> a1 = action.block(0,0,m_action_dim_per_robot,1);

            // r(0,0) = -1.0f * ( (s1-s2).transpose() * m_Q * (s1-s2) + a1.transpose() * m_R * a1);
			// r(0,0) = -1.0f * (s1-s2).transpose() * m_Q * (s1-s2);
   			// r(1,0) = -1.0f * r(0,0); 

            float dist = (s1-s2).norm();
            if (dist < m_dist) {
            	r(0,0) = 1.0;
            	r(1,0) = 0.0;
            } else { 
            	r(0,0) = 0.0;
            	r(1,0) = 0.0;
            }

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
			r(1,0) = 1 - r(0,0);
            return r;
        }


        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }
		
};
