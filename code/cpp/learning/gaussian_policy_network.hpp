

#pragma once 
#include <eigen3/Eigen/Dense>
#include "policy_network.hpp"
#include "network.hpp"

class GaussianPolicyNetwork : public PolicyNetwork {

	public: 
		Eigen::Matrix<float,-1,1> eval(
			Problem * problem, Eigen::Matrix<float,-1,1> encoding, int robot, std::default_random_engine & gen) override {
			
			int robot_action_dim = problem->m_action_idxs[robot].size();

			Eigen::Matrix<float,-1,1> action(robot_action_dim); 

			auto distribution = m_phi.eval(encoding);
			// auto mu = distribution.block(problem->m_action_idxs[robot][0],0,robot_action_dim,1);
			// auto sd = distribution.block(problem->m_action_idxs[robot][0],0,robot_action_dim,1).array().exp().sqrt();
			auto mu = distribution.block(0,0,robot_action_dim,1);
			auto sd = distribution.block(robot_action_dim,0,robot_action_dim,1).array().exp().sqrt();

			for (int i = 0; i < robot_action_dim; i++) {
				std::normal_distribution<float> dist(mu(i,0),sd(i,0));
				action(i,0) = dist(gen);
				action(i,0) = std::min(std::max(action(i,0),problem->m_action_lims(i,0)),problem->m_action_lims(i,1));
			}

			return action; 
		}

};