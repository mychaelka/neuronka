#include <vector>
#include <cmath>
#include <random>
#include <functional>

#pragma once

namespace nn {
    std::vector<Matrix> create_batches(const Matrix& data, size_t batch_size) {
        std::vector<Matrix> batches;
        size_t num_batches = (data.ncols() + batch_size - 1) / batch_size;

        for (size_t i = 0; i < num_batches; ++i) {
            size_t start = i * batch_size;
            size_t end = std::min(start + batch_size, data.ncols());

            std::vector<float> batch_elements;
            for (size_t row = 0; row < data.nrows(); ++row) {
                for (size_t col = start; col < end; ++col) {
                    batch_elements.push_back(data.get(row, col));
                }
            }

            batches.emplace_back(data.nrows(), end - start, batch_elements);
        }

        return batches;
    }

    void shuffle_batches(std::vector<Matrix>& inputs, std::vector<Matrix>& targets) {
        std::random_device rd;
        std::mt19937 generator(rd());
        std::vector<size_t> indices(inputs.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), generator);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (i != indices[i]) {
                std::swap(inputs[i], inputs[indices[i]]);
                std::swap(targets[i], targets[indices[i]]);
            }
        }
    }
    
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

        Matrix _weights_momentum;
        Matrix _bias_momentum;

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
         _weights_momentum(output_size, input_size),
         _bias_momentum(output_size, 1),
         _activation_function(relu), 
         _activation_derivative(relu_prime) {
        }

        Matrix& get_output() { return _output; }

        Matrix& get_weights() { return _weights; }

        Matrix& get_deltas() { return _deltas; }

        void init_weights() {
            std::random_device rd;
            std::mt19937 generator(rd());

            std::normal_distribution<float> weight_distribution(0.0f, std::sqrt(2.0f / _input_size)); // He 

            for (float& weight : _weights.data()) {
                weight = weight_distribution(generator);
            }

            std::uniform_real_distribution<float> bias_distribution(-0.1f, 0.1f);
            for (float& bias : _biases.data()) {
                bias = bias_distribution(generator);
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

            _output = _weights.dot_matrix(*_input);
            _output += _biases;
            _output.map(_activation_function);
        }

        void input(const Matrix& input) {
            _input = &input;
        }

        void compute_output_deltas(const Matrix& target_batch) {
            for (size_t k = 0; k < _output.ncols(); ++k) {
                for (size_t j = 0; j < _output.nrows(); ++j) {
                    float output_val = _output.get(j, k);
                    float target_val = target_batch.get(j, k);
                    float delta = output_val - target_val;

                    _deltas.set(j, k, delta);
                }
            }
        }

        void compute_hidden_deltas(const Matrix& next_weights, const Matrix& next_deltas) {
            #pragma omp parallel for collapse(2)
            for (size_t k = 0; k < _output.ncols(); ++k) {
                for (size_t j = 0; j < _output.nrows(); ++j) {
                    float delta_sum = 0.0f;

                    for (size_t i = 0; i < next_deltas.nrows(); ++i) {
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

            #pragma omp parallel for collapse(2)
            for (size_t j = 0; j < _weights.nrows(); ++j) {
                for (size_t i = 0; i < _weights.ncols(); ++i) {
                    float grad_sum = 0.0f;

                    for (size_t k = 0; k < _input->ncols(); ++k) {
                        grad_sum += _deltas.get(j, k) * _input->get(i, k);
                    }

                    _weights_grad.set(j, i, grad_sum);
                }
            }

            for (size_t j = 0; j < _bias_grad.nrows(); ++j) {
                float bias_grad_sum = 0.0f;

                for (size_t k = 0; k < _deltas.ncols(); ++k) {
                    bias_grad_sum += _deltas.get(j, k);
                }

                _bias_grad.set(j, 0, bias_grad_sum);
            }
        }

        void update(float learning_rate, float momentum, float decay) {
             _weights_grad += _weights * decay; // multiplied by the learning rate in the next step anyway
             _weights_momentum = (_weights_momentum * momentum) + (_weights_grad * (-learning_rate));
             _weights += _weights_momentum;

             _bias_momentum = (_bias_momentum * momentum) + (_bias_grad * (-learning_rate));
             _biases += _bias_momentum;
        }
        
        /* Applies softmax to the layer -- modifies the layer output, to be used 
        only for the last layer of network */
        void apply_softmax() {
            for (size_t j = 0; j < _output.ncols(); ++j) {
                float max_val = *std::max_element(&_output.data()[j * _output.nrows()],
                                                &_output.data()[(j + 1) * _output.nrows()]);
                float sum_exp = 0.0f;

                for (size_t i = 0; i < _output.nrows(); ++i) {
                    _output.set(i, j, std::exp(_output.get(i, j) - max_val));
                    sum_exp += _output.get(i, j);
                }

                for (size_t i = 0; i < _output.nrows(); ++i) {
                    _output.set(i, j, _output.get(i, j) / sum_exp);
                }
            }
        }

        void apply_dropout(float dropout_rate) {
            std::random_device rd;
            std::mt19937 generator(rd());
            std::bernoulli_distribution dropout(dropout_rate);

            for (float& val : _output.data()) {
                if (dropout(generator)) {
                    val = 0.0f;
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

        std::vector<Layer>& layers() { return _layers; }

        const Matrix& feed_forward(const Matrix& input, float dropout) {
            _layers[0].input(input);
            
            for (size_t i = 0; i < _layers.size(); ++i) {
                if (i == _layers.size() - 1) {
                    _layers[i].set_activation_function(identity, identity_prime);
                }

                _layers[i].output();
                if (i > 0 && i < _layers.size() - 1) {
                    _layers[i].apply_dropout(dropout);
                }

                if (i == _layers.size() - 1) {
                    _layers[i].apply_softmax();  // automatically apply softmax to last layer (not sure yet if good idea, less general)
                } else if (i + 1 < _layers.size()) {
                    _layers[i + 1].input(_layers[i].get_output());
                }
            }

            return _layers.back().get_output();
        } 


        void backward(const Matrix& target, float learning_rate, float momentum, float decay) {
            _layers.back().compute_output_deltas(target);

            for (int layer_idx = _layers.size() - 2; layer_idx >= 0; --layer_idx) {
                Layer& current_layer = _layers[layer_idx];
                Layer& next_layer = _layers[layer_idx + 1];

                current_layer.compute_hidden_deltas(next_layer.get_weights(), next_layer.get_deltas());
            }

            for (Layer& layer : _layers) {
                layer.compute_gradients();
                layer.update(learning_rate, momentum, decay);
            }
        }

        void fit(std::vector<Matrix>& inputs, std::vector<Matrix>& targets, int epochs, float learning_rate, 
                 float dropout, float momentum, float decay) {
            for (int epoch = 0; epoch < epochs; ++epoch) {
                shuffle_batches(inputs, targets);
                float total_loss = 0.0f;
                float adapt_learning_rate = learning_rate * (1 / (1 + decay * epoch));

                for (size_t batch_idx = 0; batch_idx < inputs.size(); ++batch_idx) {
                    const Matrix& output = feed_forward(inputs[batch_idx], dropout);

                    total_loss += cross_entropy_loss(output, targets[batch_idx]);
                    backward(targets[batch_idx], adapt_learning_rate, momentum, decay);
                }

                if (epoch % 10 == 0) {
                    std::cout << "Epoch " << epoch << "/" << epochs
                        << ", Loss: " << total_loss << std::endl;
                }
            }
        }

        std::vector<float> predict(const Matrix& input) {
            const Matrix& output = feed_forward(input, 0.0f);
            std::vector<float> predictions;

            for (size_t i = 0; i < output.ncols(); ++i) {
                size_t max_index = 0;
                float max_value = output.get(0, i);

                for (size_t j = 1; j < output.nrows(); ++j) {
                    if (output.get(j, i) > max_value) {
                        max_index = j;
                        max_value = output.get(j, i);
                    }
                }
                predictions.push_back(max_index);
            }

            return predictions;
        }
    };
}