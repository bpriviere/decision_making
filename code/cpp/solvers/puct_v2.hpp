
#pragma once 
#include "solver.hpp"

// changes to PUCT_V1:
// - resize_node
// - expand_node

class PUCT_V2 : public Solver {

	public: 
		void set_params(Solver_Settings & solver_settings, 
				std::vector<Policy_Network_Wrapper> & policy_network_wrappers,
				Value_Network_Wrapper & value_network_wrapper) override {
			std::random_device dev;
			std::default_random_engine gen(dev());  
			g_gen = gen;
			m_num_simulations = solver_settings.num_simulations;
			m_search_depth = solver_settings.search_depth;
			m_C_exp = solver_settings.C_exp;
			m_alpha_exp = solver_settings.alpha_exp;
			m_C_pw = solver_settings.C_pw;
			m_alpha_pw = solver_settings.alpha_pw;
			m_beta_policy = solver_settings.beta_policy;
			m_beta_value = solver_settings.beta_value;
		}

		struct Node { 
			Eigen::Matrix<float,-1,1> state;
			Node* parent = nullptr; 
			Eigen::Matrix<float,-1,1> action_to_node; 
			Eigen::Matrix<float,-1,1> total_value;
			int num_visits = 0;
			std::vector<Node*> children;

			int calc_depth(){
				int depth = 0;
				Node* ptr = parent;
				while (ptr) {
					ptr = ptr->parent;
					depth += 1;
				}
				return depth;
			}

			void resize_node(Problem * problem){
				action_to_node.resize(problem->m_action_dim+1,1);
				state.resize(problem->m_state_dim,1);
				total_value = Eigen::Matrix<float,-1,1>::Zero(problem->m_num_robots,1);
			}
		};


		Solver_Result search(Problem * problem, Eigen::Matrix<float,-1,1> root_state, int turn){			

			Solver_Result solver_result;

			m_nodes.clear();
			m_nodes.reserve(m_num_simulations * (m_search_depth+1) + 1);
			m_nodes.resize(1);

			auto& root_node = m_nodes[0];
			root_node.resize_node(problem);
			root_node.state = root_state;
			Node* root_node_ptr = &root_node; 

			if (problem->is_terminal(root_state)){
				solver_result.success = false;
				return solver_result; 
			}

			for (int ii = 1; ii <= m_num_simulations; ii++){

				Node* curr_node_ptr = root_node_ptr;
				std::vector< Eigen::Matrix<float,-1,1> > rewards(m_search_depth+1);
				std::vector<Node*> path(m_search_depth+1);

				for (int d = 0; d <= m_search_depth; d++){
					int robot_turn = (d + turn) % problem->m_num_robots;
					Node* child_node_ptr;
					if (is_expanded(curr_node_ptr)){
						child_node_ptr = best_child(curr_node_ptr,robot_turn);
					} else {
						child_node_ptr = expand_node(problem,curr_node_ptr); 
					}
					path[d] = curr_node_ptr;
					rewards[d] = powf(problem->m_gamma,d)*problem->normalized_reward(
						curr_node_ptr->state,child_node_ptr->action_to_node.block(0,0,problem->m_action_dim,1));
					curr_node_ptr = child_node_ptr; 
				}
				rewards[m_search_depth] = powf(problem->m_gamma,m_search_depth) * default_policy(problem,curr_node_ptr);
				path[m_search_depth] = curr_node_ptr; 

				for (int d = 0; d <= m_search_depth; d++){
					path[d]->num_visits += 1;
					path[d]->total_value += calc_value(rewards,d,m_search_depth+1,problem->m_gamma,problem->m_num_robots);
				}
			};

			solver_result.success = true;
			solver_result.best_action = most_visited(root_node_ptr,0)->action_to_node; 
			solver_result.child_distribution = export_child_distribution(problem);
			solver_result.tree = export_tree(problem);
			solver_result.value = root_node_ptr->total_value / root_node_ptr->num_visits;
			return solver_result;
		}


		Eigen::Matrix<float,-1,1> calc_value(std::vector<Eigen::Matrix<float,-1,1>> rewards, int start_depth, int total_depth, float gamma, int num_robots){
			Eigen::Matrix<float,-1,1> value = Eigen::Matrix<float,-1,1>::Zero(num_robots,1);
			for (int d = start_depth; d < total_depth; d++){
				value = value + rewards[d] * powf(gamma,d); 
			}
			return value;
		}


		Node* select_node(Problem* problem,Node* node_ptr,int robot_turn){
			while ( !problem->is_terminal(node_ptr->state) ){
				if ( is_expanded(node_ptr) ){
					node_ptr = best_child(node_ptr,robot_turn);
				} else {
					return node_ptr;
				}
			}
			return node_ptr;
		}


		Node* best_child(Node* node_ptr,int robot_turn){
			Node* result = nullptr;
			float bestValue = -1.0f;
			for (Node* c : node_ptr->children) {
				float value = c->total_value(robot_turn) / c->num_visits + m_C_exp*sqrtf(powf(node_ptr->num_visits,m_alpha_exp)/c->num_visits);
				if (value > bestValue) {
					bestValue = value;
					result = c;
				}
			}
			return result;
		}


		Node* most_visited(Node* node_ptr,int robot_turn){
			Node* result = nullptr;
			int mostVisits = 0;
			for (Node* c : node_ptr->children) {
				if (c->num_visits > mostVisits) {
					mostVisits = c->num_visits;
					result = c;
				}
			}
			return result;
		}


		bool is_expanded(Node* node_ptr){
			int max_children = ceil(m_C_pw*(powf(node_ptr->num_visits, m_alpha_pw)));
			return int(node_ptr->children.size()) > max_children;
		}


		Node* expand_node(Problem * problem,Node* parent_node_ptr){
			m_nodes.resize(m_nodes.size() + 1);
			Eigen::Matrix<float,-1,1> action(problem->m_action_dim+1,1);
			action.block(0,0,problem->m_action_dim,1) = problem->sample_action(g_gen);
			action(problem->m_action_dim) = problem->sample_timestep(g_gen,problem->m_timestep); 
			auto next_state = problem->step(parent_node_ptr->state,action.block(0,0,problem->m_action_dim,1),action(problem->m_action_dim));
			auto& child_node = m_nodes[m_nodes.size()-1];
			child_node.parent = parent_node_ptr;
			child_node.resize_node(problem);
			child_node.action_to_node = action;
			child_node.state = next_state;
			parent_node_ptr->children.push_back(&child_node);
			return &child_node;
		}


		Eigen::Matrix<float,-1,1> default_policy(Problem * problem,Node* node_ptr){
			Eigen::Matrix<float,-1,1> value = Eigen::Matrix<float,-1,1>::Zero(problem->m_num_robots,1);
			Eigen::Matrix<float,-1,1> curr_state = node_ptr->state; 
			int depth = 0; 
			while (! problem->is_terminal(curr_state) && depth < m_search_depth) 
			{
				auto action = problem->sample_action(g_gen);
				auto next_state = problem->step(curr_state,action,problem->m_timestep);
				float discount = powf(problem->m_gamma,depth); 
				value += discount * problem->normalized_reward(curr_state,action);
				curr_state = next_state;
				depth += 1;
			}
			return value; 
		}


		void backup(Node* node_ptr,Eigen::Matrix<float,-1,1> value){
			do {
				node_ptr->num_visits += 1;
				node_ptr->total_value += value;
				node_ptr = node_ptr->parent;
			} while (node_ptr != nullptr);
		}


		Eigen::MatrixXf export_tree(Problem * problem){
			Eigen::MatrixXf tree(m_nodes.size(), problem->m_state_dim + 1);
			for (int ii = 0; ii < m_nodes.size(); ++ii) {
				tree.row(ii).head(problem->m_state_dim) = m_nodes[ii].state.array();
				
				int parentIdx = -1;
				if (!(ii == 0)){
					parentIdx = m_nodes[ii].parent - &m_nodes[0];
				}
				tree(ii,problem->m_state_dim) = parentIdx;
			}
			return tree; 
		}


		Eigen::MatrixXf export_child_distribution(Problem * problem){
			Node* root_node_ptr = &m_nodes[0];
			Eigen::MatrixXf child_distribution(int(root_node_ptr->children.size()), problem->m_action_dim + 1);
			int count = 0 ;			
			for (Node* c : root_node_ptr->children) {
				// child_distribution.row(count).head(problem->m_action_dim) = c->action_to_node.block(0,0,problem->m_action_dim,1);
				child_distribution.row(count).head(problem->m_action_dim) = c->action_to_node.head(problem->m_action_dim);
				child_distribution(count,problem->m_action_dim) = c->num_visits;
				count = count + 1;
			}
			return child_distribution;
		}


	private: 
		int m_num_simulations;
		int m_search_depth;
		float m_C_exp;
		float m_alpha_exp;
		float m_C_pw;
		float m_alpha_pw;
		float m_beta_policy;
		float m_beta_value;
		std::vector<Node> m_nodes;

};
