#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <exception>
#include <algorithm>

#pragma once 

namespace nn {
    /* Linear algebra */ 
    class Matrix {
    
    private:

        size_t _nrows;
        size_t _ncols;
        std::vector<float> _data;

        void check_bounds(size_t row, size_t col) const {
            if (row >= _nrows || col >= _ncols) {
                throw std::out_of_range("Matrix indices are out of bounds");
            }
        }

        void consistency_check() {
            if (_nrows * _ncols != _data.size()) {
                throw std::length_error("Number of elements is not a multiple of rows and cols");
            }
        }

    public:

        Matrix(size_t nrows, size_t ncols) : 
            _nrows(nrows), 
            _ncols(ncols), 
            _data(nrows * ncols, 0.0f) { consistency_check(); }
        
        Matrix(size_t nrows, size_t ncols, std::vector<float> data) : 
            _nrows(nrows), 
            _ncols(ncols), 
            _data(data) { consistency_check(); }
        
        Matrix(const std::vector<std::vector<float>>& data) :
            _nrows(data.size()), 
            _ncols(data.empty() ? 0 : data[0].size()), 
            _data() {

                _data.reserve(_nrows * _ncols);
                for (const auto& row : data) {
                    if (row.size() != _ncols) {
                        throw std::invalid_argument("Inconsistent row sizes in input data");
                    }
                    _data.insert(_data.end(), row.begin(), row.end());
                }

                consistency_check();
            }

        size_t nrows() const { return _nrows; }
        size_t ncols() const { return _ncols; }

        std::vector<float>& data() { return _data; }
        const std::vector<float>& data() const { return _data; }

        float get(size_t row, size_t col) const {
            check_bounds(row, col);
            return _data[row * _ncols + col];
        }

        void set(size_t row, size_t col, float value) {
            check_bounds(row, col);
            _data[row * _ncols + col] = value;
        }

        void zero() {
            for (float& elem : _data) {
                elem = 0.0f;
            }
        }

        /* Operators */
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
            if (_nrows != other.nrows() || (other.ncols() != 1 && _ncols != other.ncols())) {
                throw std::length_error("Matrices are not of compatible shape or other matrix cannot be broadcast.");
            }

            /* Broadcasting -- because bias is a vector of input_size x 1  */
            if (other.ncols() == 1) {
                for (size_t i = 0; i < _nrows; ++i) {
                    for (size_t j = 0; j < _ncols; ++j) {
                        _data[i * _ncols + j] += other.get(i, 0);
                    }
                }
            } else {
                for (size_t i = 0; i < _nrows; ++i) {
                    for (size_t j = 0; j < _ncols; ++j) {
                        _data[i * _ncols + j] += other.get(i, j);
                    }
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

        /* Methods */
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
        
        /* Transpose -- creates new matrix */
        Matrix transpose() const {
            Matrix transposed(_ncols, _nrows);
            for (size_t i = 0; i < _nrows; ++i) {
                for (size_t j = 0; j < _ncols; ++j) {
                    transposed.set(j, i, this->get(i, j));
                }
            }
            return transposed;
        }


        template <typename Func>
        void map(Func f) {
            for (float& elem : _data) {
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

        void normalize() {
            float sum = 0.0f;
            size_t total_elements = _data.size();

            for (const float& value : _data) {
                sum += value;
            }

            float mean = sum / total_elements;

            float variance_sum = 0.0f;
            for (const float& value : _data) {
                variance_sum += (value - mean) * (value - mean);
            }

            float variance = variance_sum / total_elements;
            float std_dev = std::sqrt(variance);

            if (std_dev == 0.0f) {
                std_dev = 1.0f;
            }

            for (float& value : _data) {
                value = (value - mean) / std_dev;
            }
        }
    };
    
    /* Activation functions */
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

    /* One-hot encode individual label */
    std::vector<float> one_hot_label(int label) {
        std::vector<float> one_hot_result;

        for (int i = 0; i < 10; ++i) {
            i == label ? one_hot_result.push_back(1) : one_hot_result.push_back(0);
        }

        return one_hot_result;
    }

    /* One-hot encode multiple labels */
    std::vector<std::vector<float>> one_hot_all_labels(const std::vector<float>& labels) {
        std::vector<std::vector<float>> one_hot_labels;
        for (int label : labels) {
            one_hot_labels.push_back(one_hot_label(static_cast<int>(label)));
        }
        return one_hot_labels;
    }

    /* Loss function */
    float cross_entropy_loss(const Matrix& outputs, const Matrix& targets) {
        float total_loss = 0.0f;
        float epsilon = 1e-8;

        for (size_t k = 0; k < outputs.ncols(); ++k) {
            for (size_t i = 0; i < outputs.nrows(); ++i) {
                float p = std::max(epsilon, outputs.get(i, k));
                float t = targets.get(i, k);
                total_loss -= t * std::log(p);
            }
        }

        return total_loss;
    }
}
