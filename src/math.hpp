#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>
#include <exception>


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
        Vector(size_t size) : _size(size), _elements(size) {}  // initializes _elements with 0s
        Vector(std::vector<float> elements) : _size(elements.size()), _elements(elements) {}
        
        size_t size() const { return _size; }

        const std::vector<float>& elements() const { return _elements; }  // returns reference to vector elements

        // operators
        float& operator[](size_t index) { return _elements[index]; }
        
        const float& operator[](size_t index) const { return _elements[index]; }
        
        Vector operator+(const Vector &other) const {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            Vector out = Vector(this->size());
            for (size_t i = 0; i < this->size(); ++i) out[i] = (*this)[i] + other[i];
            return out;
        }

        void operator+=(const Vector &other) {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            for (size_t i = 0; i < this->size(); ++i) (*this)[i] += other[i];
        }

        Vector operator-(const Vector &other) const {
            if (this->size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }

            Vector out = Vector(this->size());
            for (size_t i = 0; i < this->size(); ++i) out[i] = (*this)[i] - other[i];
            return out;
        }

        void operator-=(const Vector &other) {
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
        float dot_product(const Vector &other) {
            if (_elements.size() != other.size()) {
                throw std::length_error("Vectors u and v are not of the same length.");
            }
            
            float result = 0.0;
            
            for (size_t i = 0; i < other.size(); i++) {
                result += (*this)[i] * other[i];
            }

            return result;
        }
 
    };
    
    class Matrix {
        void transpose();

    };

    // matrix-vector multiplication? If I want to do batch processing, instead of a vector of weights, I will have a matrix of weights 
    std::vector<std::vector<float>> matrix_dot_product();
    std::vector<float> matrix_dot_vector();
    std::vector<float> vector_dot_matrix();

    
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

    std::vector<float> softmax(const std::vector<float> &input) {
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
    float cross_entropy_loss(const std::vector<float> &output, const std::vector<int> &one_hot_label) {
        float loss = 0.0f;
        float epsilon = 1e-8;  // prevent log(0)

        for (size_t i = 0; i < output.size(); ++i) {
            float p = std::max(epsilon, output[i]);
            loss += one_hot_label[i] * std::log(p);
        }

        return -loss;
    }
}
