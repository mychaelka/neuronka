#include <vector>
#include <cmath>
#include <random>
#include <functional>

#pragma once

// TODO: scale the input data

namespace nn {
    class Layer {
    
    private:
    
        size_t _output_size;
        size_t _input_size;
        
        const Matrix* _input;
        Matrix _output;
        Matrix _weights;
        Matrix _biases;

        Matrix _weights_delta;
        Matrix _bias_delta;

        std::function<float(float)> _activation_function;
        std::function<float(float)> _activation_derivative;
    
    public:
        
        Layer(size_t output_size, size_t input_size)
         : _output_size(output_size),
         _input_size(input_size),
         _input(nullptr),
         _output(output_size, 1),
         _weights(output_size, input_size),
         _biases(output_size, 1),
         _weights_delta(output_size, input_size),
         _bias_delta(output_size, 1),
         _activation_function(identity), 
         _activation_derivative(identity_prime) {
        }

        const Matrix& get_output() const {
            return _output;
        }

        void init_weights(float min_value = 0.0f, float max_value = 1.0f) {
            std::random_device rd;
            std::mt19937 generator(rd());

            //std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / _input_size));
            std::uniform_real_distribution<float> distribution(min_value, max_value);

            for (float& weight : _weights.elements()) {
                weight = distribution(generator);
            }

            for (float& bias : _biases.elements()) {
                bias = distribution(generator);
            }
        }

        void set_activation_function(std::function<float(float)> activation_function, 
                                    std::function<float(float)> activation_derivative) {
            _activation_function = activation_function;
            _activation_derivative = activation_derivative;
        }

        void output() {
            if (_input == nullptr) {
                throw std::runtime_error("Input not set for layer");
            }
            
            _output = _weights.dot_matrix(*_input); // creates new matrix
            _output += _biases; // modifies _output directly
            _output.map(_activation_function); // modifies _output directly
        }

        void input(const Matrix& input) {
            _input = &input;
        }
    };


    class MLP {
    
    private:

        std::vector<size_t> _neurons;
        std::vector<Layer> _layers;
 
    public:

        MLP(std::initializer_list<size_t> layer_sizes) {
            
            auto it = layer_sizes.begin();
            size_t input_size = *it++;

            for (; it != layer_sizes.end(); ++it) {
                size_t output_size = *it;
                Layer layer(output_size, input_size);

                layer.init_weights();
                _layers.push_back(layer);

                input_size = output_size;
            }
        }

        const std::vector<Layer>& get_layers() const { return _layers; }
        std::vector<Layer>& get_layers() { return _layers; }

        /* Matrix feed_forward(const Matrix& input) {
            Matrix current_output = input;
            
            for (Layer& layer : _layers) {
                layer.set_input(current_output);
                layer.output();
                current_output = layer.get_output();
            }

            return current_output;
        }
 */
        /* void feed_input(const std::vector<float>& input_vector) {
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

        void backward_prop(const std::vector<float>& target, float learning_rate) {
            _layers.back().compute_output_deltas(target);

            for (int i = _layers.size() - 2; i > 0; --i) {
                _layers[i].compute_hidden_deltas(_layers[i + 1]);
            }

            for (size_t i = 1; i < _layers.size(); ++i) {
                _layers[i].update_weights(_layers[i - 1].get_outputs(), learning_rate);
            }
        }
        

        std::vector<float> predict() const {
            return _layers.back().get_outputs();
        }


        float compute_loss(const std::vector<float>& target) const {
            return cross_entropy_softmax(_layers.back().get_outputs(), target);
        } */
        
        float accuracy();
    };
}


void train(nn::MLP& network, const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& targets, int epochs, float learning_rate) {
    
    /* for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;

        for (size_t i = 0; i < inputs.size(); ++i) {
            network.feed_input(inputs[i]);
            network.feed_forward();

            float loss = network.compute_loss(targets[i]);
            total_loss += loss;

            network.get_layers().back().compute_output_deltas(targets[i]);

            for (int layer_idx = network.get_layers().size() - 2; layer_idx > 0; --layer_idx) {
                network.get_layers()[layer_idx].compute_hidden_deltas(network.get_layers()[layer_idx + 1]);
            }

            for (size_t layer_idx = 1; layer_idx < network.get_layers().size(); ++layer_idx) {
                std::vector<float> prev_outputs = (layer_idx == 1) ? inputs[i] : network.get_layers()[layer_idx - 1].get_outputs();
                network.get_layers()[layer_idx].update_weights(prev_outputs, learning_rate);
            }
        }

        std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Loss: " << total_loss / inputs.size() << std::endl;
    } */
}