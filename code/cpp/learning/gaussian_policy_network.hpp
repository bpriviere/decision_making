

#pragma once 
#include <eigen3/Eigen/Dense>
#include "policy_network.hpp"
#include "network.hpp"

class GaussianPolicyNetwork : public PolicyNetwork {

	public: 
		Eigen::Matrix<float,-1,1> eval(
			Problem * problem, Eigen::Matrix<float,-1,1> encoding, std::default_random_engine & gen) override {
			
			int action_dim = int(problem->m_action_dim / problem->m_num_robots);

			Eigen::Matrix<float,-1,1> action(action_dim); 

			auto distribution = m_phi.eval(encoding);
			auto mu = distribution.block(0,0,action_dim,1);
			auto sd = distribution.block(action_dim,0,action_dim,1).array().exp().sqrt();

			for (int i = 0; i < action_dim; i++) {
				std::normal_distribution<float> dist(mu(i,0),sd(i,0));
				action(i,0) = dist(gen);
			}

			return action; 
		}

};