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

    //std::cout << "Calculated cross entropy loss: " << nn::cross_entropy_loss(output, labels_class) << std::endl;
    //std::cout << "Calculated cross entropy loss for perfect classification: " << nn::cross_entropy_loss(output_perfect, labels_class) << std::endl;


    nn::Matrix matrix = nn::Matrix(5, 1);
    matrix.set(0, 0, 1);
    matrix.set(1, 0, -1);
    matrix.set(2, 0, 5);
    matrix.set(4, 0, -7);
    matrix.print_readable();
    //matrix.map([](float x) { return x * x; });
    matrix.map(nn::relu);
    matrix.print_readable();


    nn::MLP mlp({784, 8, 16, 9}, 1);

    int epochs = 100;
    float learning_rate = 0.01;    

    // Real data 
    std::vector<std::vector<float>> inputs = parse_input("./data/fashion_mnist_train_vectors.csv");
    std::vector<float> num_labels = parse_labels("./data/fashion_mnist_train_labels.csv");
    std::vector<std::vector<float>> labels = nn::one_hot_all_labels(num_labels);

    nn::Matrix label_matrix = nn::Matrix(labels).transpose();
    nn::Matrix input_matrix = nn::Matrix(inputs).transpose();
    std::cout << input_matrix.nrows() << " ";
    std::cout << input_matrix.ncols() << std::endl;

    //label_matrix.print_readable();
    std::cout << label_matrix.nrows() << " ";
    std::cout << label_matrix.ncols() << std::endl;

    //input_matrix.print_readable();

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
}