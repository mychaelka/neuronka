#include <vector>
#include <cmath>
#include <random>

#pragma once

// class neuron: inputs, weights, bias, activation
// scaling the data? 
// TODO: start with individual examples, then extend to batch processing

namespace nn {

    class Neuron {
        // constructor should initialize weights (they can be computed by the initWeights method?) and take number of weights as an argument
        // the number of weights = number of neurons in the previous layer
    private:
        Vector _weights;
        size_t _n_weights;
        float _bias;
        float _output;
    
    public:
        Neuron(size_t n_weights) : _n_weights(n_weights), _weights(n_weights) {
            init_weights();
        }

        void activate(const std::vector<float> &inputs) {
           // this->output = nn::dot_product(weights, inputs) + bias;
        }

        void print_weights() {
            for (size_t i = 0; i < _n_weights; ++i) {
                std::cout << _weights[i] << ", ";
            }

            std::cout << std::endl;
        }

        void init_weights() {
            std::random_device rd;
            std::mt19937 generator(rd());

            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / _n_weights));

            for (size_t i = 0; i < _n_weights; ++i) {
                _weights[i] = distribution(generator);
            }
        }
    };


    class Layer {
    private:
        std::vector<Neuron> neurons;
    
    public:
        std::vector<float> get_outputs() const;
        std::vector<Neuron> &get_neurons() const;
    };


    class MLP {
    private:
        std::vector<Layer> layers;
    
    public:
        void feed_forward() {
            for (int i = 1; i < this->layers.size(); ++i) { // for each layer
                for (Neuron neuron : this->layers[i].get_neurons()) { //
                    neuron.activate(this->layers[i-1].get_outputs());
                }
            }
        }

        void backward_prop();
        void predict();
        float accuracy();
    };
}