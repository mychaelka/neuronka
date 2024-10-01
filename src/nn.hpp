#include <vector>

// class neuron: inputs, weights, bias, activation

class Neuron {
public:
    std::vector<float> weights;
    int n_weights;
    float bias;
    float output;
};


class Layer {
public:
    std::vector<Neuron> neurons;
};


class MLP {
public:
    std::vector<Layer> layers;
};