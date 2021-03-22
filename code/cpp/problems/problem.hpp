
#include <vector>
#include <random>

// All internal functions of problem need to be overloaded

class Problem
{
public:
    int m_state_dim{};
    Eigen::MatrixXd m_state_lims{};
    int m_action_dim{};
    Eigen::MatrixXd m_action_lims{};
    int m_num_robots{};

    // forward propagate dynamics 
    Eigen::MatrixXd step(
        Eigen::MatrixXd state,
        Eigen::MatrixXd action)
    {
        Eigen::MatrixXd next_state;
        return next_state;
    }

    // calculate rewards  
    Eigen::MatrixXd rewards(
        Eigen::MatrixXd state,
        Eigen::MatrixXd action)
    {
        Eigen::MatrixXd reward;
        return reward;
    }

    // stop condition
    bool isTerminal(
        Eigen::MatrixXd state)
    {
        return true;
    }

    // initialize state 
    Eigen::MatrixXd initialize(
        std::default_random_engine& generator
        )
    {
        Eigen::MatrixXd state;
        return state;
    }

    // action sample  
    Eigen::MatrixXd sample_action(
        std::default_random_engine& generator
        )
    {
        Eigen::MatrixXd action;
        return action;
    }
};