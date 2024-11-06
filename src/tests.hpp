#include <iostream>
#include <vector>
#include <cassert>
#include "math.hpp"
#include "parser.hpp"
#include "nn.hpp"

void tests() {

    // Bad input file, should fail
    std::vector<std::vector<float>> bad_input = parse_input("nonexistent_file.csv");
    std::vector<std::vector<float>> good_input = parse_input("./data/fashion_mnist_train_vectors.csv");

    //for (int i = 0; i < good_input[0].size(); i++) {
    //    std::cout << good_input[0][i] << std::endl;
    //}

    std::vector<float> labels = parse_labels("./data/fashion_mnist_train_labels.csv");
    //for (int i = 0; i < labels.size(); i++) {
    //    std::cout << labels[i] << ", ";
    //}

    // softmax
    std::vector<float> vec = {1.8f, 0.9f, 0.68f};
    std::vector<float> vec_output = nn::softmax(vec);

    for (std::vector<float>::iterator it = vec_output.begin(); it != vec_output.end(); ++it) {
        std::cout << *it << std::endl;
    }

    // cross entropy
    std::vector<float> output = {0.2f, 0.6f, 0.05f, 0.1f, 0.05f};
    std::vector<float> output_perfect = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<int> labels_class = {0, 1, 0, 0, 0};

    std::cout << "Calculated cross entropy loss: " << nn::cross_entropy_loss(output, labels_class) << std::endl;
    std::cout << "Calculated cross entropy loss for perfect classification: " << nn::cross_entropy_loss(output_perfect, labels_class) << std::endl;

    // Vector class operators
    std::vector<float> u = {1, 2, 3, 4, 5};
    nn::Vector vec1 = nn::Vector(u);
    vec1[4] = 8;
    nn::Vector vec2 = nn::Vector(u);
    nn::Vector sum = vec1 + vec2;

    std::cout << "Sum of two vectors: " << std::endl;
    sum.print_readable();

    vec1 += vec2;
    std::cout << "+= modifies given vector: " << std::endl;
    vec1.print_readable();

    //nn::Matrix wrong_prod = matrix1.dot_matrix(matrix3);

    //nn::Neuron neuron = nn::Neuron(5);
    //neuron.print_weights();

    //nn::MLP mlp = nn::MLP(7, 4, 4, 7, 9);

    //size_t j = 0;
    //for (nn::Layer& layer : mlp.get_layers()) {
    //    if (j == mlp.num_layers() - 1) {
    //        layer.set_activation_function(nn::sigmoid, nn::sigmoid_prime);
    //    } else {
    //        layer.set_activation_function(nn::relu, nn::relu_prime);
    //    }
    //    ++j;
    //}


    //std::vector<float> input = {1, 2, 255, 255, 0, 0, 6};
    //mlp.feed_input(input);

    //std::cout << "Inputs fed into the network: " << std::endl;

    //std::vector<float> outputs = mlp.get_layers()[0].get_outputs();
    //for (float output : outputs) {
    //    std::cout << output << ", ";
    //}

    //std::cout << std::endl;

    //mlp.feed_forward(); // so far no activations are implemented

    //std::cout << "Outputs of the network: " << std::endl;

    //std::vector<float> out = mlp.predict();
    //for (float output : out) {
    //    std::cout << output << ", ";
    //}

    //std::cout << std::endl;

    //std::cout << "Current network configuration: " << std::endl;
    //size_t i = 0;
    //for (nn::Layer layer : mlp.get_layers()) {
    //    std::cout << "Layer: " << i << std::endl;
    //    for (nn::Neuron neuron : layer.get_neurons()) {
    //        neuron.print_weights();
    //        std::cout << "Output: " << neuron.output() << std::endl;
    //    }

    //    std::cout << '\n';
    //    ++i;
    //}

    std::cout << "BACKPROPAGATION TEST" << std::endl;
    nn::MLP mlp1(3, 4, 2);

    mlp1.get_layers()[1].set_activation_function(nn::relu, nn::relu_prime);     // Hidden layer
    mlp1.get_layers()[2].set_activation_function(nn::sigmoid, nn::sigmoid_prime); // Output layer

    // Sample input and target
    std::vector<float> input1 = {0.5, 0.1, -0.3};
    std::vector<float> target1 = {1.0, 0.0};  // Example target

    for (int epoch = 0; epoch < 1000; ++epoch) {
        mlp1.feed_input(input1);
        mlp1.feed_forward();
        mlp1.backward_prop(target1, 0.01);
    }


    std::vector<float> output1 = mlp1.predict();
    std::cout << "Output after training: ";
    for (float val : output1) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}