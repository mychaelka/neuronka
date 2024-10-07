#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <exception>

#pragma once 

namespace nn {
    /*
      Linear algebra
    */ 
    
    // std::vector does not have the typical vector operations defined, it is just a container 
    class Vector {  // TODO: switch to template
    private:
        size_t _size;
        std::vector<float> _elements;
    
    public:
        // TODO: write vector initializer with elements list
        Vector() : _size(0), _elements() {}
        Vector(size_t size) : _size(size), _elements(size) {}
        Vector(std::vector<float> elements) : _size(elements.size()), _elements(elements) {}
        // Constructor that takes an initializer list
        Vector(std::initializer_list<float> elements) : _size(elements.size()), _elements(elements) {}
        
        size_t size() const { return this->_size; }

        const std::vector<float>& elements() const { return _elements; }  // returns reference to vector elements

        void set_weights(std::vector<float>& new_weights) {
            _elements = new_weights;
        }

        // operators
        float& operator[](size_t index) { return _elements[index]; }
        
        const float& operator[](size_t index) const { return _elements[index]; }
        
        Vector operator+(const Vector& other) const {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            Vector out = Vector(this->size());
            for (size_t i = 0; i < this->size(); ++i) out[i] = (*this)[i] + other[i];
            return out;
        }

        void operator+=(const Vector& other) {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            for (size_t i = 0; i < this->size(); ++i) (*this)[i] += other[i];
        }

        Vector operator-(const Vector& other) const {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            Vector out = Vector(this->size());
            for (size_t i = 0; i < this->size(); ++i) out[i] = (*this)[i] - other[i];
            return out;
        }

        void operator-=(const Vector& other) {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            for (size_t i = 0; i < this->size(); ++i) (*this)[i] -= other[i];
        }

        Vector operator*(const float num) const {
            Vector out = Vector(this->size());
            for (size_t i = 0; i < this->size(); ++i) out[i] = (*this)[i] * num;
            return out;
        }

        void operator*=(const float num) {
            for (size_t i = 0; i < this->size(); ++i) (*this)[i] *= num;
        }

        // vector * vector?

        // methods
        float dot_product(const Vector& other) {
            if (_elements.size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }
            
            float result = 0.0;
            
            for (size_t i = 0; i < other.size(); i++) {
                result += (*this)[i] * other[i];
            }

            return result;
        }

        void print_readable() {
            std::cout << '(';

            for (float elem : this->_elements) {
                std::cout << elem << ", ";
            }
            std::cout << ')' << std::endl;
        }
 
    };
    
    class Matrix {
    private:
        size_t _nrows;
        size_t _ncols;
        std::vector<Vector> _elements;  //std::vector<Vector> or std::vector<float>? 
    
    public:
        Matrix(size_t nrows, size_t ncols) : _nrows(nrows), _ncols(ncols), _elements(nrows, ncols) {}
        Matrix(std::vector<Vector> elements) : _nrows(elements.size()), _ncols(elements[0].size()), _elements(elements) {}

        // operators
        size_t nrows() const { return this->_nrows; }
        size_t ncols() const { return this->_ncols; }

        float get(size_t row, size_t col) const {
            return this->_elements[row][col];
        }

        void set(size_t row, size_t col, float value) {
            this->_elements[row][col] = value;
        }

        Matrix operator+(const Matrix& other) const {
            if (this->nrows() != other.nrows() || this->ncols() != other.ncols()) {
                throw std::length_error("Matrices are not of the same shape.");
            }

            Matrix out = Matrix(this->nrows(), this->ncols());
            for (size_t i = 0; i < this->nrows(); ++i) {
                for (size_t j = 0; j < this->ncols(); ++j) {
                    out.set(i, j, this->get(i, j) + other.get(i, j));
                }
            }
            return out;
        }

        void operator+=(const Matrix& other) {
            if (this->nrows() != other.nrows() || this->ncols() != other.ncols()) {
                throw std::length_error("Matrices are not of the same shape.");
            }

            for (size_t i = 0; i < this->nrows(); ++i) {
                for (size_t j = 0; j < this->ncols(); ++j) {
                    (*this).set(i, j, this->get(i, j) + other.get(i, j));
                }
            }
        }

        Matrix operator*(const float num) const {
            Matrix out = Matrix(this->nrows(), this->ncols());
            for (size_t i = 0; i < this->nrows(); ++i) {
                for (size_t j = 0; j < this->ncols(); ++j) {
                    out.set(i, j, this->get(i, j) * num);
                }
            }
            return out;
        }

        void operator*=(const float num) {
            for (size_t i = 0; i < this->nrows(); ++i) {
                for (size_t j = 0; j < this->ncols(); ++j) {
                    (*this).set(i, j, this->get(i, j) * num);
                }
            }
        }

        // methods
        Matrix dot_matrix(const Matrix& other) const {  // this matrix muse have the same number of columns as other has rows
            if (this->ncols() != other.nrows()) {
                throw std::length_error("Matrices are not of compatible shape.");
            }

            float current_cell;

            Matrix out = Matrix(this->nrows(), other.ncols());
            for (size_t k = 0; k < this->nrows(); ++k) {  // for every row of first matrix
                for (size_t i = 0; i < this->ncols(); ++i) {  // for every element in row of first matrix
                    current_cell = 0;
                    for (size_t j = 0; j < other.nrows(); ++j) {  // for every element in col of second matrix
                        current_cell += this->get(k,j) * other.get(j, i);
                    }
                    
                    out.set(k, i, current_cell);
                }
            }

            return out;
        }

        void print_readable() {
            for (size_t i = 0; i < this->nrows(); ++i) {
                std::cout << '|';
                for (size_t j = 0; j < this->ncols(); ++j) {
                    std::cout << this->get(i, j) << ", ";
                }

                std::cout << "|\n";
            }
        }

        //void transpose();

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

    std::vector<int> one_hot_label(int label) {
        std::vector<int> one_hot_result;

        for (int i = 0; i < 10; ++i) {
            i == label ? one_hot_result.push_back(1) : one_hot_result.push_back(0);
        }

        return one_hot_result;
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
}
