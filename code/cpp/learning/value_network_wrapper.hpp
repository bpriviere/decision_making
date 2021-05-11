
#pragma once
#include <iostream>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "value_network.hpp"
#include "deterministic_value_network.hpp"

// Strategy Pattern: https://stackoverflow.com/questions/41220046/is-it-possible-to-change-a-c-objects-class-after-instantiation

class Value_Network_Wrapper
{
	public:
		ValueNetwork* value_network;
		bool valid;

		Value_Network_Wrapper(){
			valid = false;
		}

		void initialize(std::string value_name){
			value_network = new ValueNetwork();
			if (value_name == "deterministic"){
				value_network = new DeterministicValueNetwork(); 
			}
		}

		void addLayer(Eigen::MatrixXf weight, Eigen::MatrixXf bias){
			value_network->m_phi.addLayer(weight,bias);
			valid = true;
		}
};