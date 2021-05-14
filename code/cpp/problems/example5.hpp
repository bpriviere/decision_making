
#pragma once 

// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example5 : public Problem { 
	
	public:
		float m_r_max; 
		float m_r_min; 
		float m_m1; 
		float m_m2; 
		float m_c1; 
		float m_c2; 

		void set_params(Problem_Settings & problem_settings) override 
		{
			int m_state_dim_per_robot = 1;
			int m_action_dim_per_robot = 1;
			m_num_robots = 2;
			m_state_dim = m_num_robots * m_state_dim_per_robot;
			m_action_dim = m_num_robots * m_action_dim_per_robot;

			problem_settings.state_lims.resize(m_state_dim,2);
			problem_settings.action_lims.resize(m_action_dim,2);
			problem_settings.init_lims.resize(m_state_dim,2);

			m_timestep = problem_settings.timestep;
			m_gamma = problem_settings.gamma;
			m_r_max = problem_settings.r_max;
			m_r_min = problem_settings.r_min;
			m_state_lims = problem_settings.state_lims; 
			m_action_lims = problem_settings.action_lims; 
			m_init_lims = problem_settings.init_lims;
			m_m1 = problem_settings.m1; 
			m_m2 = problem_settings.m2;
			m_c1 = problem_settings.c1; 
			m_c2 = problem_settings.c2; 

			std::uniform_real_distribution<double> dist(0,1.0f); 

		}


		Eigen::Matrix<float,-1,1> step(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action,
			float timestep) override
		{
			Eigen::Matrix<float,-1,1> next_state(m_state_dim,1);
			next_state(0,0) = state(0,0) + timestep * (m_m1 - m_c1 * action(1,0) * state(1,0));
			next_state(1,0) = state(1,0) + timestep * (m_m2 - m_c2 * action(0,0) * state(0,0));
			return next_state;
		}


		Eigen::Matrix<float,-1,1> reward(
			Eigen::Matrix<float,-1,1> state,
			Eigen::Matrix<float,-1,1> action) override
		{ 
			Eigen::Matrix<float,-1,1> r(m_num_robots,1);
			r(0,0) = -1*((1-action(1,0))*state(1,0) - (1-action(0,0))*state(0,0));
			r(1,0) = -1 * r(0,0);
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


		bool is_valid(Eigen::Matrix<float,-1,1> state) override {
			bool stateInBounds = (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
			return stateInBounds;
		}
		
};
