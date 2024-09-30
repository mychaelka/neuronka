#include <vector>
#include <iostream>
#include <cmath>

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

// No need to implement tanh, it can already be found in std::tanh