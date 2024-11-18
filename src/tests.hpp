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

    // Bad input file, should fail
    std::vector<std::vector<float>> bad_input = parse_input("nonexistent_file.csv");
    std::vector<std::vector<float>> good_input = parse_input("./data/fashion_mnist_train_vectors.csv");

    nn::Matrix matrix = nn::Matrix(5, 1);
    matrix.set(0, 0, 1);
    matrix.set(1, 0, -1);
    matrix.set(2, 0, 5);
    matrix.set(4, 0, -7);
    matrix.print_readable();
    //matrix.map([](float x) { return x * x; });
    matrix.map(nn::relu);
    //matrix.print_readable();

    // MNIST
    std::vector<std::vector<float>> train_inputs = parse_input("./data/fashion_mnist_train_vectors.csv");
    std::vector<float> num_labels = parse_labels("./data/fashion_mnist_train_labels.csv");
    std::vector<std::vector<float>> train_labels = nn::one_hot_all_labels(num_labels);

    nn::Matrix label_matrix = nn::Matrix(train_labels).transpose();
    nn::Matrix input_matrix = nn::Matrix(train_inputs).transpose();
    std::cout << input_matrix.nrows() << " ";
    std::cout << input_matrix.ncols() << std::endl;

    std::cout << label_matrix.nrows() << " ";
    std::cout << label_matrix.ncols() << std::endl;

    //mlp.feed_forward(input_matrix);
    //nn::Matrix out = mlp.layers().back().get_output();
    

    /* train(mlp, inputs, labels, epochs, learning_rate);

    std::cout << "Predictions after training:" << std::endl;
    for (const auto& input : inputs) {
        mlp.feed_input(input);
        mlp.feed_forward();
        std::vector<float> output = mlp.predict();

        for (float val : output) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    } */

   /* XOR */
    nn::Matrix xor_input_matrix(2, 4, {0, 0, 1, 1, 0, 1, 1, 0});
    nn::Matrix xor_target_matrix(2, 4, {1, 0, 1, 0, 0, 1, 0, 1});

    nn::MLP xor_network({2, 4, 2}, 4); // 2 input neurons, 2 hidden neurons, 1 output neurons, batch size = 4

    // Train the network
    int epochs = 1000;
    float learning_rate = 0.05f;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        const nn::Matrix& outputs = xor_network.feed_forward(xor_input_matrix);
        
        float loss = nn::cross_entropy_loss(outputs, xor_target_matrix);
        
        xor_network.backward(xor_target_matrix, learning_rate);

        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss << std::endl;
        }
    }

    std::vector<float> predictions = xor_network.predict(xor_input_matrix);
    write_predictions("output.csv", predictions);

}