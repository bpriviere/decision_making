
#pragma once
#include <eigen3/Eigen/Dense>
#include "network.hpp"

// internal functions of problem need to be overloaded
class PolicyNetwork
{
    public:
        FeedForwardNetwork m_phi;
        virtual ~PolicyNetwork() { }

        virtual Eigen::Matrix<float,-1,1> eval(
            Problem * problem, Eigen::Matrix<float,-1,1> encoding, int robot, std::default_random_engine & gen) 
        {
            Eigen::Matrix<float,-1,1> result;
            return result;
        }
};