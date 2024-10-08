#include <vector>
#include <cmath>
#include <random>
#include <functional>

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

        void set_output(float output) { _output = output; }

        void set_inner_potential(const nn::Vector& inputs) {
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
        std::function<float(float)> _activation_function;
    
    public:
        Layer(size_t n_neurons, size_t n_weights_per_neuron)
         : _activation_function(identity) {
            for (size_t i = 0; i < n_neurons; ++i) {
                if (n_weights_per_neuron > 0) {
                    _neurons.emplace_back(n_weights_per_neuron);
                } else {
                    Neuron input_neuron(0);
                    _neurons.push_back(input_neuron);
                }
            }
        }

        std::vector<float> get_outputs() const {
            std::vector<float> outputs;
            for (const Neuron& neuron : get_neurons()) {
                outputs.push_back(neuron.output());
            }

            return outputs;
        }

        const std::vector<Neuron>& get_neurons() const { return _neurons; }
        std::vector<Neuron>& get_neurons() { return _neurons; }

        void set_activation_function(std::function<float(float)> activation_function) {
            _activation_function = activation_function;
        }


        void activate() {
            for (Neuron& neuron : _neurons) {
                float activated_output = _activation_function(neuron.output());
                neuron.set_output(activated_output);
            }
        }
    };


    class MLP {
    private:
        std::vector<Layer> _layers;
        size_t _n_layers;

        template<typename... Sizes>
        void create_layers(size_t input_size, size_t hidden_size, Sizes... sizes) {
            _layers.emplace_back(hidden_size, input_size);
            create_layers(hidden_size, sizes...);
        }

        void create_layers(size_t hidden_size, size_t output_size) {
            _layers.emplace_back(output_size, hidden_size);
        }

    
    public:
        template<typename... Sizes>
        MLP(Sizes... sizes) {
            static_assert(sizeof...(sizes) >= 2, "At least two sizes (input and output layers) are required.");
            
            auto get_first = [](auto first, auto... rest) {
                return first; 
            };

            size_t input_size = get_first(sizes...);
            _layers.emplace_back(input_size, 0);
            create_layers(sizes...);
        }

        const std::vector<Layer>& get_layers() const { return _layers; }
        std::vector<Layer>& get_layers() { return _layers; }

        size_t num_layers() { return _n_layers; }

        void feed_input(std::vector<float>& input_vector) {
            size_t i = 0;
            
            for (Neuron& input_neuron : _layers[0].get_neurons()) { // for every neuron in input layer...
                input_neuron.set_output(input_vector[i]);
                ++i;
            }
        }

        void feed_forward() {
            for (int i = 1; i < _layers.size(); ++i) {
                for (Neuron& neuron : _layers[i].get_neurons()) {
                    neuron.set_inner_potential(_layers[i-1].get_outputs());
                }

                _layers[i].activate();
            }
        }

        void backward_prop();
        
        std::vector<float> predict() const {
            return _layers[_layers.size() - 1].get_outputs();
        }
        
        float accuracy();
    };
}