
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include <math.h>       /* sin , cos , tan */
#include "problem.hpp"

class Example12 : public Problem { 
	
	public:
		float m_r_max; 
		float m_r_min; 
		float m_c1; 
		float m_c2; 
		float m_R; 
		float m_desired_distance;
		float m_tf;

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
			m_tf = problem_settings.tf; 
			m_gamma = problem_settings.gamma;
			m_r_max = problem_settings.r_max;
			m_r_min = problem_settings.r_min;
			m_state_lims = problem_settings.state_lims; 
			m_action_lims = problem_settings.action_lims; 
			m_init_lims = problem_settings.init_lims;
			m_c1 = problem_settings.c1; 
			m_c2 = problem_settings.c2; 
			m_R = problem_settings.R; 
			m_desired_distance = problem_settings.desired_distance;

			std::uniform_real_distribution<double> dist(0,1.0f); 

		}


		Eigen::Matrix<float,-1,1> step(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			float timestep) override
		{
			Eigen::Matrix<float,-1,1> next_state(m_state_dim,1);
			next_state(0,0) = state(0,0) + timestep * m_c1 * action(0,0);
			next_state(1,0) = state(1,0) + timestep * m_c1 * action(1,0);
			next_state(2,0) = state(2,0) + timestep * m_c2 * sin(state(4,0));
			next_state(3,0) = state(3,0) + timestep * m_c2 * cos(state(4,0));
			next_state(4,0) = state(4,0) + timestep * m_c2 / m_R * action(2,0);
			next_state(5,0) = state(5,0) + timestep;

			// wrap angles
			// next_state(4,0) = fmod(next_state(4,0) + M_PI, 2*M_PI) - M_PI;
			return next_state;
		}


		Eigen::Matrix<float,-1,1> reward(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action) override
		{ 
			// Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			// r(0,0) = 0.5;
			// if (is_captured(state)) {
			// 	r(0,0) = 0.0;
			// }
			// return r;

			return normalized_reward(state,action);

		}


		Eigen::Matrix<float,-1,1> normalized_reward(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action) override
		{
			// Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			// r = reward(state,action);
			// r = r.cwiseMin(m_r_max).cwiseMax(m_r_min);
			// r.array() = (r.array() - m_r_min) / (m_r_max - m_r_min);
			// r(1,0) = 1 - r(0,0);

			// Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			// if (is_terminal(state)) {
			// 	float r1 = (float) (state.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
			// 		(state.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all();
			// 	float r2 = (float) (state.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() && 
			// 		(state.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all();
			// 	float r3 = 1.0f; 

			// 	r(0,0) = (0.1 * r1 + 0.1 * (1-r2) + 0.8 * r3);
			// 	r(1,0) = 1 - r(0,0);
			// } else {
			// 	r(0,0) = 0.0f ;
			// 	r(1,0) = 0.0f;
			// };

			// Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			// float r1 = (float) (state.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
			// 	(state.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all();
			// float r2 = (float) (state.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() && 
			// 	(state.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all();
			// float r3 = 1.0f; 

			// r(0,0) = (0.1 * r1 + 0.1 * (1-r2) + 0.8 * r3);
			// r(1,0) = 1 - r(0,0);

			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r(0,0) = 0.0;
			r(1,0) = 0.0;
			if (is_captured(state)) {
				r(0,0) = state(5,0) / m_tf;
				r(1,0) = 1 - r(0,0);
			} else if (state(5,0) > m_tf) {
				r(0,0) = 1.0;
				r(1,0) = 0.0;
			} else if ( !(
				(state.block(0,0,2,1).array() >= m_state_lims.block(0,0,2,1).array()).all() && 
				(state.block(0,0,2,1).array() <= m_state_lims.block(0,1,2,1).array()).all() )) {
				r(0,0) = -1;
			} else if ( !(
				(state.block(2,0,2,1).array() >= m_state_lims.block(2,0,2,1).array()).all() &&
				(state.block(2,0,2,1).array() <= m_state_lims.block(2,1,2,1).array()).all() )) {
				r(1,0) = -1;
			}
			return r;
		}


		bool is_valid(Eigen::Matrix<float,-1,1> state) override {
			bool stateInBounds = (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
			return stateInBounds;
		}

		bool is_terminal(Eigen::Matrix<float,-1,1> state) override 
		{
			// return !is_valid(state);
			return (!is_valid(state)) || is_captured(state);
		}

		bool is_captured(Eigen::Matrix<float,-1,1> state) {
			return (state.block(0,0,2,1) - state.block(2,0,2,1)).norm() < m_desired_distance;
		}


	    Eigen::Matrix<float,-1,1> policy_encoding(Eigen::Matrix<float,-1,1> state, int robot) override {
	        // return state;

	    	Eigen::Matrix<float,-1,1> new_state(2,1); 
	    	new_state(0,0) = state(0,0) - state(2,0);
	    	new_state(1,0) = state(1,0) - state(3,0);

	    	Eigen::Matrix<float,2,2> R;
	    	R << 
	    		cos(state(4,0)),-sin(state(4,0)),
	    		sin(state(4,0)), cos(state(4,0)); 

    		new_state = R * new_state;
	        return new_state;
	    }

	    Eigen::Matrix<float,-1,1> value_encoding(Eigen::Matrix<float,-1,1> state) override {
	        // return state;

	    	Eigen::Matrix<float,-1,1> new_state(2,1); 
	    	new_state(0,0) = state(0,0) - state(2,0);
	    	new_state(1,0) = state(1,0) - state(3,0);

	    	Eigen::Matrix<float,2,2> R;
	    	R << 
	    		cos(state(4,0)),-sin(state(4,0)),
	    		sin(state(4,0)), cos(state(4,0)); 

    		new_state = R * new_state;
	        return new_state;
	    }

		
};
