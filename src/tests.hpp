#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include "math.hpp"
#include "parser.hpp"
#include "nn.hpp"


void write_predictions(const std::string& filename, const std::vector<float>& predictions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    for (const auto& prediction : predictions) {
        file << static_cast<int>(prediction) << "\n"; // Ensure output is an integer
    }

    file.close();
}


void tests() {

    // MNIST
    std::vector<std::vector<float>> train_inputs = parse_input("./data/fashion_mnist_train_vectors.csv");
    std::vector<std::vector<float>> test_inputs = parse_input("./data/fashion_mnist_test_vectors.csv");

    std::vector<float> train_labels = parse_labels("./data/fashion_mnist_train_labels.csv");
    std::vector<std::vector<float>> train_targets = nn::one_hot_all_labels(train_labels);

    std::vector<float> test_labels = parse_labels("./data/fashion_mnist_test_labels.csv");
    std::vector<std::vector<float>> test_targets = nn::one_hot_all_labels(test_labels);

    nn::Matrix train_input_matrix = nn::Matrix(train_inputs).transpose();
    nn::Matrix train_target_matrix = nn::Matrix(train_targets).transpose();

    nn::Matrix test_input_matrix = nn::Matrix(test_inputs).transpose();
    nn::Matrix test_target_matrix = nn::Matrix(test_targets).transpose();
    
    size_t train_size = 60000;
    size_t test_size = 10000;
    size_t input_size = 784;
    size_t num_classes = 10;
    size_t batch_size = 128;

    nn::MLP mnist_network({input_size, 32, 64, num_classes}, batch_size);

    train_input_matrix.normalize();
    test_input_matrix.normalize();

    int epochs = 100;
    float learning_rate = 0.01f;

    auto input_batches = nn::create_batches(train_input_matrix, batch_size);
    auto target_batches = nn::create_batches(train_target_matrix, batch_size);

    nn::train(mnist_network, input_batches, target_batches, epochs, learning_rate);

    std::vector<float> train_predictions = mnist_network.predict(train_input_matrix);
    std::vector<float> test_predictions = mnist_network.predict(test_input_matrix);

    write_predictions("train_predictions.csv", train_predictions);
    write_predictions("test_predictions.csv", test_predictions);
}