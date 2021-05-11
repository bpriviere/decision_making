
#pragma once
#include <eigen3/Eigen/Dense>
#include <vector>

struct Layer {
	Eigen::MatrixXf weight;
	Eigen::MatrixXf bias;
	};

class FeedForwardNetwork {
	
	public:	
		void addLayer(Eigen::MatrixXf weight, Eigen::MatrixXf bias){
			m_layers.push_back({weight, bias});
			}

		Eigen::VectorXf eval( Eigen::VectorXf& input){
			assert(m_layers.size() > 0);
			Eigen::VectorXf result = input;
			for (size_t i = 0; i < m_layers.size()-1; ++i) {
				auto& l = m_layers[i];
				result = relu(l.weight * result + l.bias);
			}
			auto& l = m_layers.back();
			result = l.weight * result + l.bias;
			return result;
			}

		size_t sizeIn(){
			assert(m_layers.size() > 0);
			return m_layers[0].weight.cols();
		}

		size_t sizeOut(){
			assert(m_layers.size() > 0);
			return m_layers.back().bias.size();
		}

		bool valid(){
			return m_layers.size() > 0;
		}

	private:
		std::vector<Layer> m_layers;
	
		Eigen::MatrixXf relu( Eigen::MatrixXf m){
			return m.cwiseMax(0);
		}

};