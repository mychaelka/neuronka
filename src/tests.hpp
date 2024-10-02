#include <iostream>
#include <vector>
#include "math.hpp"
#include "parser.hpp"

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
}