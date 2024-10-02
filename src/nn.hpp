#include <vector>

// class neuron: inputs, weights, bias, activation

namespace nn {

    class Neuron {
    private:
        std::vector<float> weights;
        int n_weights;
        float bias;
        float output;
    
    public:
        void activate(std::vector<float> inputs) {
            this->output = nn::dot_product(weights, inputs) + bias;
        }
        void init_weights();
    };


    class Layer {
    private:
        std::vector<Neuron> neurons;
    
    public:
        std::vector<Neuron> get_neurons();
    };


    class MLP {
    private:
        std::vector<Layer> layers;
    
    public:
        void feed_forward() {
            for (int i = 1; i < this->layers.size(); ++i) { // for each layer
                for (Neuron neuron : this->layers[i].get_neurons()) { //
                    neuron.activate(this->layers[i-1].get_neurons());
                }
            }
        }

        void backward_prop();
        void predict();
        float accuracy();
    };
}