
#pragma once
#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "policy_network.hpp"
#include "gaussian_policy_network.hpp"

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Policy_Network_Wrapper
{
	public:
		PolicyNetwork* policy_network;
		bool valid;

		Policy_Network_Wrapper(){
			valid = false;
		}

		void initialize(std::string policy_name){
			policy_network = new PolicyNetwork();
			if (policy_name == "gaussian"){
				policy_network = new GaussianPolicyNetwork(); 
			}
		}

		void addLayer(Eigen::MatrixXf weight, Eigen::MatrixXf bias){
			policy_network->m_phi.addLayer(weight,bias);
			valid = true;
		}
};