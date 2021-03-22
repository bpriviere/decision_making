
// #include "../problems/example1.hpp"

struct Node { 
	Eigen::Matrix<float,2,1> state;
	Node* parent = nullptr; 
	Eigen::Matrix<float,2,1> action_to_node; 
	Eigen::Matrix<float,1,1> total_value;
	int num_visits = 0;
	std::vector<Node*> children;

	int calc_depth()
		{
			int depth = 0;
			Node* ptr = parent;
			while(ptr) {
				ptr = ptr->parent;
				depth += 1;
			}
			return depth;
		}
};

class PUCT {
	public: 
		Example1 m_problem; 
		int m_num_nodes; 
		int m_search_depth;
		std::vector<Node> m_nodes;
		float m_C_exp;
		float m_alpha_exp;
		float m_C_pw;
		float m_alpha_pw;
		float m_beta_policy;
		float m_beta_value;

		PUCT() // default constructor 
		{
			m_num_nodes = 1000; 
			m_search_depth = 10;
			m_C_exp = 1.0f;
			m_alpha_exp = 0.25f; 
			m_C_pw = 2.0f; 
			m_alpha_pw = 0.5f; 
			m_beta_policy = 0.0f; 
			m_beta_value = 0.0f; 
		}


		Node search(
			Eigen::Matrix<float,2,1> root_state)
			{
				m_nodes.clear();
				m_nodes.reserve(m_num_nodes+1);
				m_nodes.resize(1);

				auto& root_node = m_nodes[0];
				root_node.state = root_state;
				Node* root_node_ptr = &root_node; 

				if (m_problem.is_terminal(root_state)){
					return root_node; 
				}

				for (int ii = 0; ii < m_num_nodes; ii++){
					int robot_turn = ii % m_problem.m_num_robots; 
					Node* parent_node_ptr = select_node(root_node_ptr,robot_turn); 
					Node* child_node_ptr = expand_node(parent_node_ptr);
					auto value = default_policy(child_node_ptr);
					backup(child_node_ptr,value); 
				};

				return root_node;
			}


		Node* select_node(
			Node* node_ptr,
			int robot_turn)
			{
				while ( !m_problem.is_terminal(node_ptr->state) ){
					if ( is_expanded(node_ptr) ){
						node_ptr = best_child(node_ptr,robot_turn);
					} else {
						return node_ptr;
					}
				}
				return node_ptr;
			}


		Node* best_child(
			Node* node_ptr,
			int robot_turn)
		{
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


		bool is_expanded(
			Node* node_ptr)
			{
				int max_children = ceil(m_C_pw*(powf(node_ptr->num_visits, m_alpha_pw)));
				return node_ptr->children.size() > max_children;
			}


		Node* expand_node(
			Node* parent_node_ptr)
			{
				auto action = m_problem.sample_action();
				auto next_state = m_problem.step(parent_node_ptr->state,action);
				m_nodes.resize(m_nodes.size() + 1);
				auto& child_node = m_nodes[m_nodes.size()-1];
				child_node.parent = parent_node_ptr;
				child_node.action_to_node = action;
				parent_node_ptr->children.push_back(&child_node);
				return &child_node;
			}


		Eigen::Matrix<float,1,1> default_policy(
			Node* node_ptr)
			{
				Eigen::Matrix<float,1,1> value;
				float total_discount = 0.0;
				int depth = node_ptr->calc_depth();
				Eigen::Matrix<float,2,1> curr_state = node_ptr->state; 
				while (! m_problem.is_terminal(curr_state) && depth < m_search_depth) 
				{
					auto action = m_problem.sample_action();
					auto next_state = m_problem.step(curr_state,action);
					float discount = powf(m_problem.m_gamma,depth); 
					value += discount * m_problem.normalized_reward(curr_state,action);
					total_discount += discount; 
					curr_state = next_state;
					depth += 1;
				}
				if (total_discount > 0){
					value  = value / total_discount;
				}
				return value; 
			}


		void backup(
			Node* node_ptr, 
			Eigen::Matrix<float,1,1> value)
			{
				do {
					node_ptr->num_visits += 1;
					node_ptr->total_value += value;
					node_ptr = node_ptr->parent;
				} while (node_ptr != nullptr);
			}
};
