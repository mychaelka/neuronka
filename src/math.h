#include <vector>

// Linear algebra
float dot_product(std::vector<float> u, std::vector<float> v);

// Activation functions
float sigmoid(float x);

float sigmoid_prime(float x);

float relu(float x);

float relu_prime(float x);

float tanh(float x);

float tanh_prime(float x);