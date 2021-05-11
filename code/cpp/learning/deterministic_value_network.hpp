
#pragma once 
#include <eigen3/Eigen/Dense>
#include "value_network.hpp"
#include "network.hpp"

class DeterministicValueNetwork : public ValueNetwork {

	public: 
		Eigen::Matrix<float,-1,1> eval(
			Problem * problem, Eigen::Matrix<float,-1,1> encoding, std::default_random_engine & gen) override {
			Eigen::Matrix<float,-1,1> value(problem->m_num_robots); 
            value = m_phi.eval(encoding);
			return value; 
		}

};