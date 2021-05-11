
#pragma once
#include <eigen3/Eigen/Dense>
#include "network.hpp"

// internal functions of value need to be overloaded
class ValueNetwork
{
    public:
        FeedForwardNetwork m_phi;
        virtual ~ValueNetwork() { }

        virtual Eigen::Matrix<float,-1,1> eval(
            Problem * problem, Eigen::Matrix<float,-1,1> encoding, std::default_random_engine & gen) 
        {
            Eigen::Matrix<float,-1,1> result;
            return result;
        }
};