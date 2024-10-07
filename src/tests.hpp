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

    // Matrix
    nn::Matrix matrix1 = nn::Matrix(4, 3);
    nn::Matrix matrix2 = nn::Matrix(3, 2);
    nn::Matrix matrix3 = nn::Matrix(5, 2);

    matrix1.set(0, 0, 1);
    matrix1.set(0, 2, 3);
    matrix1.set(1, 1, 6);
    matrix1.set(0, 0, 1);
    matrix1.set(0, 3, 4);
    matrix1.set(2, 0, 5);
    matrix1.set(2, 0, 2);
    matrix1.set(3, 1, 2);
    matrix1.set(3, 2, 7);
    matrix1.print_readable();

    matrix2.set(0, 1, 1);
    matrix2.set(1, 1, 3);
    matrix2.set(2, 1, 6);
    matrix2.set(0, 0, 1);
    matrix2.set(2, 2, 4);
    matrix2.set(2, 0, 5);
    matrix2.set(0, 2, 2);
    matrix2.print_readable();

    std::cout << '\n' << "Product of two matrices: " << '\n';
    nn::Matrix prod_matrix = matrix1.dot_matrix(matrix2);
    prod_matrix.print_readable();

    //nn::Matrix wrong_prod = matrix1.dot_matrix(matrix3);

    nn::Neuron neuron = nn::Neuron(5);
    neuron.print_weights();

    nn::Layer layer = nn::Layer(3);
    std::vector<nn::Neuron> neurons = layer.get_neurons();

    for (nn::Neuron neuron : neurons) {
        neuron.print_weights();
    }
}