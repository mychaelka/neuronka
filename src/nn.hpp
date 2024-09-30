#include <vector>

// class neuron: inputs, weights, bias, activation

class Neuron {
public:
    std::vector<float> weights;
    float bias;
};


class Layer {
public:
    std::vector<Neuron> neurons;
};


class MLP {
public:
    std::vector<float> weights;
    float bias;
};