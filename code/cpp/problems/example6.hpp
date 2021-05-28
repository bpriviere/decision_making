
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example6 : public Problem { 
	
	public:
		Eigen::Matrix<float,2,2> m_Fc;
		Eigen::Matrix<float,2,2> m_Bc;
		Eigen::Matrix<float,2,2> m_I; 
		Eigen::Matrix<float,2,2> m_Q;
		Eigen::Matrix<float,2,2> m_R;  
		Eigen::Matrix<float,-1,1> m_desired_state;  
		std::vector<Eigen::Matrix<float,2,2>> m_obstacles;
		float m_r_max; 
		float m_r_min; 
		float m_state_control_weight;
		float m_desired_distance;

		void set_params(Problem_Settings & problem_settings) override 
		{
			m_state_dim = problem_settings.state_dim;
			m_action_dim = problem_settings.action_dim;
			m_num_robots = problem_settings.num_robots;
			m_state_idxs = problem_settings.state_idxs;
			m_action_idxs = problem_settings.action_idxs;
			m_desired_state = problem_settings.desired_state;

			// problem_settings.state_lims.resize(m_state_dim,2);
			// problem_settings.action_lims.resize(m_action_dim,2);
			// problem_settings.init_lims.resize(m_state_dim,2);

			m_timestep = problem_settings.timestep;
			m_gamma = problem_settings.gamma;
			m_r_max = problem_settings.r_max;
			m_r_min = problem_settings.r_min;
			m_state_control_weight = problem_settings.state_control_weight;
			m_state_lims = problem_settings.state_lims; 
			m_action_lims = problem_settings.action_lims; 
			m_init_lims = problem_settings.init_lims;
			m_obstacles = problem_settings.obstacles; 
			m_desired_distance = problem_settings.desired_distance;

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
			next_state = Fd * state + Bd * action; 
			return next_state;
		}


		Eigen::Matrix<float,-1,1> reward(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action) override
		{ 
			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r = -1 * ((state-m_desired_state).transpose() * m_Q * (state-m_desired_state) + action.transpose() * m_R * action); 
			return r;
		}


		Eigen::Matrix<float,-1,1> normalized_reward(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action) override
		{
			// Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			// r = reward(state,action);
			// r = r.cwiseMin(m_r_max).cwiseMax(m_r_min);
			// r.array() = (r.array() - m_r_min) / (m_r_max - m_r_min);
			// return r;

			Eigen::Matrix<float,-1,1> r(m_num_robots,1);\
			r(0,0) = 0.0f; 
			if ((state - m_desired_state).norm() < m_desired_distance) {
				r(1,0) = 1.0f;
			}
			return r;
		}



		bool is_valid(Eigen::Matrix<float,-1,1> state) override {

			bool stateInBounds = (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
			bool collisionFree = true;

			for (Eigen::Matrix<float,2,2> obstacle : m_obstacles){
				if (
					state(0,0) >= obstacle(0,0) && 
					state(0,0) <= obstacle(0,1) &&
					state(1,0) >= obstacle(1,0) &&
					state(1,0) <= obstacle(1,1) ){
			
					collisionFree = false; 
					break; 
				}
			}
			
			return stateInBounds && collisionFree;
		}

		bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
		{
			return !is_valid(state);
		}
		
};
