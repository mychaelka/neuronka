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

        Matrix _deltas;

        Matrix _weights_grad;
        Matrix _bias_grad;

        std::function<float(float)> _activation_function;
        std::function<float(float)> _activation_derivative;
    
    public:
        
        Layer(size_t output_size, size_t input_size, size_t batch_size)
         : _output_size(output_size),
         _input_size(input_size),
         _input(nullptr),  // input_size x 1
         _output(output_size, batch_size),
         _weights(output_size, input_size),
         _biases(output_size, 1),
         _deltas(output_size, batch_size),
         _weights_grad(output_size, input_size),
         _bias_grad(output_size, 1),
         _activation_function(relu), 
         _activation_derivative(relu_prime) {
        }

        Matrix& get_output() { return _output; }

        Matrix& get_weights() { return _weights; }

        void init_weights(float min_value = 0.0f, float max_value = 1.0f) {
            std::random_device rd;
            std::mt19937 generator(rd());

            //std::normal_distribution<float> distribution(0.0f, std::sqrt(2.0f / _input_size));
            std::uniform_real_distribution<float> distribution(min_value, max_value);

            for (float& weight : _weights.data()) {
                weight = distribution(generator);
            }

            for (float& bias : _biases.data()) {
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

            if (_output_size != _output.nrows()) {
                throw std::length_error("Output size is not equal to number of neurons in layer");
            }
        }

        void input(const Matrix& input) {
            _input = &input;
        }

        void compute_output_deltas(const Matrix& target_batch) {
            for (size_t k = 0; k < _output.ncols(); ++k) { // for every example
                for (size_t j = 0; j < _output.nrows(); ++j) { // for every neuron
                    float output_val = _output.get(j, k);
                    float target_val = target_batch.get(j, k);
                    float delta = output_val - target_val;

                    _deltas.set(j, k, delta);
                }
            }
        }

        void compute_hidden_deltas(const Matrix& next_weights, const Matrix& next_deltas) {
            for (size_t k = 0; k < _output.ncols(); ++k) { // for every example
                for (size_t j = 0; j < _output.nrows(); ++j) { // for every neuron in this layer
                    float delta_sum = 0.0f;

                    for (size_t i = 0; i < next_deltas.nrows(); ++i) { // for every neuron in the next layer
                        delta_sum += next_weights.get(i, j) * next_deltas.get(i, k);
                    }

                    float output_val = _output.get(j, k);
                    _deltas.set(j, k, delta_sum * _activation_derivative(output_val));
                }
            }
        }

        void compute_gradients() {
            _weights_grad.zero();
            _bias_grad.zero();

            for (size_t j = 0; j < _weights.nrows(); ++j) {        // for each output neuron
                for (size_t i = 0; i < _weights.ncols(); ++i) {    // for each input neuron
                    float grad_sum = 0.0f;

                    for (size_t k = 0; k < _input->ncols(); ++k) {  // for each example in the batch
                        grad_sum += _deltas.get(j, k) * _input->get(i, k);
                    }

                    _weights_grad.set(j, i, grad_sum);
                }
            }

            for (size_t j = 0; j < _bias_grad.nrows(); ++j) {  // for each output neuron
                float bias_grad_sum = 0.0f;

                for (size_t k = 0; k < _deltas.ncols(); ++k) {  // for each example in the batch
                    bias_grad_sum += _deltas.get(j, k);
                }

                _bias_grad.set(j, 0, bias_grad_sum);
            }
        }

        
        /* Applies softmax to the layer -- modifies the layer output, to be used 
        only for the last layer of network */
        void apply_softmax() {
            for (size_t j = 0; j < _output.ncols(); ++j) {
                float max_val = _output.get(0, j);
                for (size_t i = 1; i < _output.nrows(); ++i) {
                    max_val = std::max(max_val, _output.get(i, j));
                }

                float sum_exp = 0.0f;
                for (size_t i = 0; i < _output.nrows(); ++i) {
                    float exp_val = std::exp(_output.get(i, j) - max_val);
                    _output.set(i, j, exp_val);
                    sum_exp += exp_val;
                }
                
                for (size_t i = 0; i < _output.nrows(); ++i) {
                    _output.set(i, j, _output.get(i, j) / sum_exp);
                }
            }
        }

    };


    class MLP {
    
    private:

        std::vector<size_t> _neurons;
        std::vector<Layer> _layers;
 
    public:

        MLP(std::initializer_list<size_t> layer_sizes, size_t batch_size) {
            
            auto it = layer_sizes.begin();
            size_t input_size = *it++;

            for (; it != layer_sizes.end(); ++it) {
                size_t output_size = *it;
                Layer layer(output_size, input_size, batch_size);

                layer.init_weights();
                _layers.push_back(layer);

                input_size = output_size;
            }
        }

        const std::vector<Layer>& layers() const { return _layers; }
        std::vector<Layer>& layers() { return _layers; }

        const Matrix& feed_forward(const Matrix& input) {
            _layers[0].input(input);
            
            for (size_t i = 0; i < _layers.size(); ++i) {
                _layers[i].output();

                if (i == _layers.size() - 1) {
                    _layers[i].apply_softmax();  // automatically apply softmax to last layer (not sure yet if good idea, less general)
                } else if (i + 1 < _layers.size()) {
                    _layers[i + 1].input(_layers[i].get_output());
                }
            }

            return _layers.back().get_output();
        } 

        std::vector<float> predict() const {}
    };


    /* void train(nn::MLP& network, const std::vector<std::vector<float>>& inputs,
           const std::vector<std::vector<float>>& targets, int epochs, float learning_rate, size_t batch_size) {
    
        const size_t num_samples = inputs.size();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            float total_loss = 0.0f;

            for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                size_t current_batch_size = std::min(batch_size, num_samples - batch_start);

                Matrix input_matrix(inputs[0].size(), current_batch_size);
                Matrix target_matrix(targets[0].size(), current_batch_size);

                for (size_t i = 0; i < current_batch_size; ++i) {
                    const auto& input = inputs[batch_start + i];
                    const auto& target = targets[batch_start + i];

                    for (size_t j = 0; j < input.size(); ++j) {
                        input_matrix.set(j, i, input[j]);  // j is input dimension, i is batch sample
                    }

                    for (size_t j = 0; j < target.size(); ++j) {
                        target_matrix.set(j, i, target[j]);  // j is target dimension, i is batch sample
                    }
                }

                const Matrix& output = network.feed_forward(input_matrix);

                float batch_loss = 0.0f;
                for (size_t i = 0; i < current_batch_size; ++i) {
                    std::vector<float> output_sample(output.nrows());
                    std::vector<float> target_sample(target_matrix.nrows());
                    
                    for (size_t j = 0; j < output_sample.size(); ++j) {
                        output_sample[j] = output.get(j, i);
                    }
                    for (size_t j = 0; j < target_sample.size(); ++j) {
                        target_sample[j] = target_matrix.get(j, i);
                    }

                    batch_loss += network.categorical_cross_entropy(output_sample, target_sample);
                }
                total_loss += batch_loss / current_batch_size;

                network.backward(target_matrix, learning_rate);
            }

            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                    << ", Loss: " << total_loss / (num_samples / batch_size) << std::endl;
        }
    } */
}