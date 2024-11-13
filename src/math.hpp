#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <exception>
#include <algorithm>

#pragma once 

namespace nn {
    /*
      Linear algebra
    */ 
    class Matrix {
    
    private:

        size_t _nrows;
        size_t _ncols;
        std::vector<float> _elements;

        void check_bounds(size_t row, size_t col) const {
            if (row >= _nrows || col >= _ncols) {
                throw std::out_of_range("Matrix indices are out of bounds");
            }
        }

    public:

        Matrix(size_t nrows, size_t ncols) : 
            _nrows(nrows), 
            _ncols(ncols), 
            _elements(nrows * ncols, 0.0f) {}
        
        Matrix(size_t nrows, size_t ncols, std::vector<float> elements) : 
            _nrows(nrows), 
            _ncols(ncols), 
            _elements(elements) {}

        void consistency_check() {
            if (_nrows * _ncols != _elements.size()) {
                throw std::length_error("Number of elements is not a multiple of rows and cols");
            }
        }

        // operators
        size_t nrows() const { return _nrows; }
        size_t ncols() const { return _ncols; }

        std::vector<float>& elements() { return _elements; }
        const std::vector<float>& elements() const { return _elements; }

        float get(size_t row, size_t col) const {
            check_bounds(row, col);
            return _elements[row * _ncols + col];
        }

        void set(size_t row, size_t col, float value) {
            check_bounds(row, col);
            _elements[row * _ncols + col] = value;
        }

        void zero() {
            for (float& elem : _elements) {
                elem = 0.0f;
            }
        }

        Matrix operator+(const Matrix& other) const {
            if (_nrows != other.nrows() || this->ncols() != other.ncols()) {
                throw std::length_error("Matrices are not of the same shape.");
            }

            Matrix result(_nrows, _ncols);
            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < _ncols; ++j) {
                    result.set(i, j, this->get(i, j) + other.get(i, j));
                }
            }
            return result;
        }

        Matrix& operator+=(const Matrix& other) {
            if (_nrows != other.nrows() || _ncols != other.ncols()) {
                throw std::length_error("Matrices are not of the same shape.");
            }

            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < _ncols; ++j) {
                    (*this).set(i, j, this->get(i, j) + other.get(i, j));
                }
            }

            return *this;
        }

        Matrix operator*(const float num) const {
            Matrix result(nrows(), ncols());
            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < _ncols; ++j) {
                    result.set(i, j, this->get(i, j) * num);
                }
            }
            return result;
        }

        Matrix& operator*=(const float num) {
            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < _ncols; ++j) {
                    (*this).set(i, j, this->get(i, j) * num);
                }
            }

            return *this;
        }

        // methods
        Matrix dot_matrix(const Matrix& other) const {
            if (_ncols != other.nrows()) {
                throw std::length_error("Matrices are not of compatible shape.");
            }

            Matrix result(_nrows, other.ncols());
            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < other.ncols(); ++j) {
                    float current_cell = 0;
                    for (size_t k = 0; k < _ncols; ++k) {
                        current_cell += this->get(i,k) * other.get(k, j);
                    }
                    
                    result.set(i, j, current_cell);
                }
            }

            return result;
        }

        template <typename Func>
        void map(Func f) {
            for (float& elem : _elements) {
                elem = f(elem);
            }
        }

        void print_readable() {
            for (size_t i = 0; i < _nrows; ++i) {
                std::cout << '|';
                for (size_t j = 0; j < _ncols; ++j) {
                    std::cout << this->get(i, j) << ", ";
                }

                std::cout << "|\n";
            }
        }
    };

    // matrix-vector multiplication? If I want to do batch processing, instead of a vector of weights, I will have a matrix of weights 
    //std::vector<std::vector<float>> matrix_dot_product();
    //std::vector<float> matrix_dot_vector();
    //std::vector<float> vector_dot_matrix();

    
    /*
      Activation functions
    */
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float sigmoid_prime(float x) {
        return sigmoid(x) * (1.0f - sigmoid(x));
    }

    float relu(float x) {
        return std::max(0.0f, x);
    }

    float relu_prime(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    float leaky_relu(float x) {
        return std::max(0.1f * x, x);
    }

    float leaky_relu_prime(float x) {
        return x > 0.0f ? 1.0f : 0.01f;
    }

    float identity(float x) {
        return x;
    }

    float identity_prime(float x) {
        return 1.0f;
    }


    std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> out;
        float exp_sum = 0.0f;

        for (size_t i = 0; i < input.size(); ++i) {
            exp_sum += std::exp(input[i]);
        }

        for (size_t j = 0; j < input.size(); ++j) {
            out.push_back(std::exp(input[j]) / exp_sum);
        }
        return out;
    }

    std::vector<float> one_hot_label(int label) {
        std::vector<float> one_hot_result;

        for (int i = 0; i < 10; ++i) {
            i == label ? one_hot_result.push_back(1) : one_hot_result.push_back(0);
        }

        return one_hot_result;
    }

    std::vector<std::vector<float>> one_hot_all_labels(const std::vector<float>& labels) {
        std::vector<std::vector<float>> one_hot_labels;
        for (int label : labels) {
            one_hot_labels.push_back(one_hot_label(static_cast<int>(label)));
        }
        return one_hot_labels;
    }

    /*
      Loss function
    */
    float cross_entropy_loss(const std::vector<float>& output, const std::vector<int>& one_hot_label) {
        float loss = 0.0f;
        float epsilon = 1e-8;  // prevent log(0)

        for (size_t i = 0; i < output.size(); ++i) {
            float p = std::max(epsilon, output[i]);
            loss += one_hot_label[i] * std::log(p);
        }

        return -loss;
    }

    float cross_entropy_softmax(const std::vector<float>& output, const std::vector<float>& target) {
        if (output.size() != target.size()) {
            throw std::invalid_argument("Logits and target size must be the same.");
        }

        float max_logit = *std::max_element(output.begin(), output.end());

        float log_sum_exp = 0.0f;
        for (float logit : output) {
            log_sum_exp += std::exp(logit - max_logit);
        }
        log_sum_exp = max_logit + std::log(log_sum_exp);

        float loss = 0.0f;
        for (size_t i = 0; i < output.size(); ++i) {
            loss -= target[i] * (output[i] - log_sum_exp);
        }

        return loss;
    }

}
