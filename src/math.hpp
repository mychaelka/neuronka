#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>

/*
    Linear algebra
*/ 

class Vector {
public:
    size_t length;
    std::vector<float> elements;

    Vector(size_t length) : elements(length) {}  // initialize zero vector
    Vector(const std::vector<float>& vec) : elements(vec) {}  // initialize with a given vector, copy constructor for std::vector
};


class Matrix {

};

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

std::vector<float> softmax(std::vector<float> input) {
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

// No need to implement tanh, it can already be found in std::tanh