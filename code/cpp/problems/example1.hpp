
#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <eigen3/Eigen/Dense>

class Example1 { 
    
    public:
        int m_state_dim;
        int m_action_dim;
        int m_num_robots;
        float m_timestep;
        float m_gamma; 
        std::default_random_engine gen;
        std::uniform_real_distribution<double> dist;
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

            std::uniform_real_distribution<double> dist(0,1.0f); 

            m_state_lims(0,0) = -1.0f;
            m_state_lims(0,1) = 1.0f;
            m_state_lims(1,0) = -1.0f;
            m_state_lims(1,1) = 1.0f;

            m_action_lims(0,0) = -1.0f;
            m_action_lims(0,1) = 1.0f;
            m_action_lims(1,0) = -1.0f;
            m_action_lims(1,1) = 1.0f;

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


        Eigen::Matrix<float,2,1> step(
            Eigen::Matrix<float,2,1> state,
            Eigen::Matrix<float,2,1> action)
            {
                Eigen::Matrix<float,2,1> next_state; 
                next_state = m_F * state + m_B * action; 
                return next_state;
            }


        Eigen::Matrix<float,1,1> reward(
            Eigen::Matrix<float,2,1> state,
            Eigen::Matrix<float,2,1> action)
            {
                Eigen::Matrix<float,1,1> r;
                r = -1 * (state.transpose() * m_Q * state + action.transpose() * m_R * action); 
                return r;
            }


        Eigen::Matrix<float,1,1> normalized_reward(
            Eigen::Matrix<float,2,1> state,
            Eigen::Matrix<float,2,1> action)
            {
                Eigen::Matrix<float,1,1> r;
                r = reward(state,action);
                float r_max = 100;
                float r_min = -1 * r_max;  
                r = r.cwiseMin(r_min).cwiseMax(r_max);                
                return (r.array() - r_min) / (r_max - r_min) ;
            }
        

        Eigen::Matrix<float,2,1> sample_state()
            {
                Eigen::Matrix<float,2,1> state; 
                for (int ii = 0; ii < m_state_dim; ii++){
                    float alpha = dist(gen); 
                    state(ii,0) = alpha * (m_state_lims(ii,1) - m_state_lims(ii,0)) + m_state_lims(ii,0);
                }
                return state;
            }


        Eigen::Matrix<float,2,1> sample_action()
            {
                Eigen::Matrix<float,2,1> action; 
                for (int ii = 0; ii < m_action_dim; ii++){
                    float alpha = dist(gen); 
                    action(ii,0) = alpha * (m_action_lims(ii,1) - m_action_lims(ii,0)) + m_action_lims(ii,0);
                }
                return action;
            } 
        

        bool is_terminal(
            Eigen::Matrix<float,2,1> state
        )
        {
            return !is_valid(state);
        }


        bool is_valid(
            Eigen::Matrix<float,2,1> state
        )
        {
            return (state.array() >= m_state_lims.col(0).array()).all() && (state.array() <= m_state_lims.col(1).array()).all();
        }
};
