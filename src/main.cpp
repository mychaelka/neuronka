#include <iostream>
#include <vector>
#include "math.hpp"
#include "parser.hpp"
#include "nn.hpp"


void mnist() {
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
    
    size_t input_size = 784;
    size_t num_classes = 10;
    size_t batch_size = 128;

    nn::MLP mnist_network({input_size, 128, 64, num_classes}, batch_size);

    train_input_matrix.normalize();
    test_input_matrix.normalize();

    int epochs = 100;
    float learning_rate = 0.001f;
    float dropout_rate = 0.4f;
    float momentum = 0.7f;
    float decay = 0.0001f;

    auto input_batches = nn::create_batches(train_input_matrix, batch_size);
    auto target_batches = nn::create_batches(train_target_matrix, batch_size);

    mnist_network.fit(input_batches, target_batches, epochs, learning_rate, dropout_rate, momentum, decay);

    std::vector<float> train_predictions = mnist_network.predict(train_input_matrix);
    std::vector<float> test_predictions = mnist_network.predict(test_input_matrix);

    write_predictions("train_predictions.csv", train_predictions);
    write_predictions("test_predictions.csv", test_predictions);
}


int main() {
    mnist();

    return 0;
}