#include <vector>
#include <cmath>
#include <random>

#pragma once

// class neuron: inputs, weights, bias, activation
// scaling the data? 
// TODO: start with individual examples, then extend to batch processing

namespace nn {

    class Neuron {
    private:
        Vector _weights;
        size_t _n_weights;
        float _bias;
        float _output;
    
    public:
        Neuron(size_t n_weights) : _n_weights(n_weights), _weights(n_weights) {
            init_weights();
        }

        size_t n_weights() const { return _n_weights; }
        const Vector& get_weights() const { return _weights; }
        void set_weights(std::vector<float>& new_weights) { 
            _weights.set_weights(new_weights);
        }

        float output() const { return _output; }

        void activate(const nn::Vector& inputs) {
            _output = _weights.dot_product(inputs) + _bias;
        }

        void print_weights() {

            std::cout << "Weights: ";

            for (size_t i = 0; i < _n_weights; ++i) {
                std::cout << _weights[i] << ", ";
            }

            std::cout << std::endl;
            std::cout << "Bias: " << _bias << std::endl;
        }

        void init_weights() {
            std::random_device rd;
            std::mt19937 generator(rd());

            std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / _n_weights));
            std::uniform_real_distribution<float> bias_distribution(-0.1f, 0.1f);

            for (size_t i = 0; i < _n_weights; ++i) {
                _weights[i] = distribution(generator);
            }

            _bias = bias_distribution(generator);
        }
    };


    class Layer {
    private:
        std::vector<Neuron> _neurons;
    
    public:
        Layer(size_t n_neurons, size_t n_weights_per_neuron) {
            for (size_t i = 0; i < n_neurons; ++i) {
                _neurons.emplace_back(n_weights_per_neuron);
            }
        }
        std::vector<float> get_outputs() const;
        const std::vector<Neuron> &get_neurons() const { return _neurons; }
    };


    class MLP {
    private:
        std::vector<Layer> _layers;
    
    public:
        MLP() {} // TODO: initializer list
        void feed_forward() {
            for (int i = 1; i < this->_layers.size(); ++i) { // for each layer
                for (Neuron neuron : this->_layers[i].get_neurons()) { //
                    neuron.activate(this->_layers[i-1].get_outputs());
                }
            }
        }

        void backward_prop();
        void predict();
        float accuracy();
    };
}