
// #include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>
#include "problem.hpp"

class Example1 : public Problem { 
    
    public:
        std::uniform_real_distribution<double> dist;
        std::default_random_engine m_gen;
        Eigen::Matrix<float,2,2> m_state_lims;  
        Eigen::Matrix<float,2,2> m_action_lims;  
        Eigen::Matrix<float,2,2> m_F;
        Eigen::Matrix<float,2,2> m_B;
        Eigen::Matrix<float,2,2> m_Q;
        Eigen::Matrix<float,2,2> m_R;  


        Example1() // default constructor
        {
            m_state_dim = 2;
            m_action_dim = 2;
            m_num_robots = 1;
            m_timestep = 0.1f;
            m_gamma = 1.0f;

            float pos_lim = 5.0f;
            float vel_lim = 1.0f;

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_state_lims(0,0) = -pos_lim;
            m_state_lims(0,1) = pos_lim;
            m_state_lims(1,0) = -pos_lim;
            m_state_lims(1,1) = pos_lim;

            m_action_lims(0,0) = -vel_lim;
            m_action_lims(0,1) = vel_lim;
            m_action_lims(1,0) = -vel_lim;
            m_action_lims(1,1) = vel_lim;

            m_F(0,0) = 1.0f;
            m_F(0,1) = 0.0f; 
            m_F(1,0) = 0.0f; 
            m_F(1,1) = 1.0f; 

            m_B(0,0) = 1.0f * m_timestep;
            m_B(0,1) = 0.0f * m_timestep; 
            m_B(1,0) = 0.0f * m_timestep; 
            m_B(1,1) = 1.0f * m_timestep; 

            m_Q(0,0) = 1.0f;
            m_Q(0,1) = 0.0f; 
            m_Q(1,0) = 0.0f; 
            m_Q(1,1) = 1.0f; 

            m_R(0,0) = 1.0f;
            m_R(0,1) = 0.0f; 
            m_R(1,0) = 0.0f; 
            m_R(1,1) = 1.0f; 
        }


        Eigen::Matrix<float,-1,1> step(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
            {
                Eigen::Matrix<float,-1,1> next_state(m_state_dim,1); 
                next_state = m_F * state + m_B * action; 
                return next_state;
            }


        Eigen::Matrix<float,-1,1> reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
            { 
                Eigen::Matrix<float,-1,1> r(m_num_robots,1);
                r = -1 * (state.transpose() * m_Q * state + action.transpose() * m_R * action); 
                return r;
            }


        Eigen::Matrix<float,-1,1> normalized_reward(
            Eigen::Matrix<float,-1,1> state,
            Eigen::Matrix<float,-1,1> action) override
            {
                Eigen::Matrix<float,-1,1> r(m_num_robots,1);
                r = reward(state,action);
                float r_max = 100;
                float r_min = -1 * r_max;  
                r = r.cwiseMin(r_max).cwiseMax(r_min);
                r.array() = (r.array() - r_min) / (r_max - r_min);
                return r;
            }
        

        Eigen::Matrix<float,-1,1> sample_state(std::default_random_engine & gen)
            {
                Eigen::Matrix<float,-1,1> state(m_state_dim,1); 
                for (int ii = 0; ii < m_state_dim; ii++){
                    float alpha = dist(gen); 
                    // float alpha = dist(m_gen); 
                    state(ii,0) = alpha * (m_state_lims(ii,1) - m_state_lims(ii,0)) + m_state_lims(ii,0);
                }
                return state;
            }


        Eigen::Matrix<float,-1,1> sample_action(std::default_random_engine & gen) override
            {
                Eigen::Matrix<float,2,1> action(m_action_dim,1); 
                for (int ii = 0; ii < m_action_dim; ii++){
                    float alpha = dist(gen); 
                    // float alpha = dist(m_gen); 
                    action(ii,0) = alpha * (m_action_lims(ii,1) - m_action_lims(ii,0)) + m_action_lims(ii,0);
                }
                return action;
            } 
        

        Eigen::Matrix<float,-1,1> initialize(std::default_random_engine & gen) override
            {
                auto state = sample_state(gen); 
                return state;
            }



        bool is_terminal(Eigen::Matrix<float,-1,1> state) override
        {
            return !is_valid(state);
        }


        bool is_valid(Eigen::Matrix<float,-1,1> state) override
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }
};
