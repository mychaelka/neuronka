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

        Matrix _weights_gradient;
        Matrix _bias_gradient;

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
         _weights_delta(output_size, input_size),
         _bias_delta(output_size, 1),
         _weights_gradient(output_size, input_size),
         _bias_gradient(output_size, 1),
         _activation_function(relu), 
         _activation_derivative(relu_prime) {
        }

        Matrix& get_output() { return _output; }

        Matrix& get_weights() { return _weights; }

        Matrix& get_deltas() { return _weights_delta; }

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
            _output += _biases; // modifies _output directly - TODO: broadcast
            _output.map(_activation_function); // modifies _output directly
        }

        void input(const Matrix& input) {
            _input = &input;
        }

        void update(float learning_rate) {
             _weights += _weights_delta * (-learning_rate);
             _biases += _bias_delta * (-learning_rate);
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

        void compute_output_delta(const Matrix& target_batch) {
            for (size_t i = 0; i < _output.nrows(); ++i) {
                for (size_t j = 0; j < _output.ncols(); ++j) {
                    float output_val = _output.get(i, j);
                    float target_val = target_batch.get(i, j);
                    _bias_delta.set(i, j, output_val - target_val);
                }
            }
        }

        void compute_hidden_delta(const Matrix& next_weights, const Matrix& next_delta) {
            for (size_t i = 0; i < _output_size; ++i) {
                for (size_t j = 0; j < _output.ncols(); ++j) {
                    float delta_sum = 0.0f;

                    for (size_t k = 0; k < next_delta.nrows(); ++k) {
                        delta_sum += next_weights.get(k, i) * next_delta.get(k, j);
                    }

                    float output_val = _output.get(i, j);
                    _bias_delta.set(i, j, delta_sum * _activation_derivative(output_val));
                }
            }
        }

        void compute_gradients() {
            _weights_delta.zero();
            _bias_delta.zero();

            for (size_t i = 0; i < _weights.nrows(); ++i) {
                for (size_t j = 0; j < _weights.ncols(); ++j) {
                    float grad_sum = 0.0f;
                    for (size_t k = 0; k < _input->ncols(); ++k) {
                        grad_sum += _bias_delta.get(i, k) * _input->get(j, k);
                    }
                    _weights_delta.set(i, j, grad_sum / _input->ncols());
                }
            }

            for (size_t i = 0; i < _bias_delta.nrows(); ++i) {
                float bias_grad_sum = 0.0f;
                for (size_t k = 0; k < _bias_delta.ncols(); ++k) {
                    bias_grad_sum += _bias_delta.get(i, k);
                }
                _bias_delta.set(i, 0, bias_grad_sum / _bias_delta.ncols());
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
                    _layers[i].apply_softmax();
                } else if (i + 1 < _layers.size()) {
                    _layers[i + 1].input(_layers[i].get_output());
                }
            }

            return _layers.back().get_output();
        }

        void backward(const Matrix& target, float learning_rate) {
            _layers.back().compute_output_delta(target);

            for (int layer_idx = _layers.size() - 2; layer_idx >= 0; --layer_idx) {
                Layer& current_layer = _layers[layer_idx];
                Layer& next_layer = _layers[layer_idx + 1];
                
                current_layer.compute_hidden_delta(next_layer.get_weights(), next_layer.get_deltas());
            }

            for (Layer& layer : _layers) {
                layer.compute_gradients();
                layer.update(learning_rate);
            }
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