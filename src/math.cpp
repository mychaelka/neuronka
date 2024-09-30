#include "math.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>

// Linear algebra
float dot_product(std::vector<float> u, std::vector<float> v) {
    
    if (u.size() != v.size()) {
        std::cerr << "Vectors u and v are not of the same length." << std::endl;
    }

    float result = 0.0;

    for (size_t i = 0; i < u.size(); i++) {
        result += u[i] * v[i];
    }

    return result;
}

// Activation functions
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

float tanh(float x) {
    return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
}

float tanh_prime(float x) {
    return tanh(x) * tanh(x);
}