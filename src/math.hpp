#include <vector>
#include <iostream>
#include <cmath>
#include <numeric>


namespace nn {
    /*
    Linear algebra
    */ 
    float dot_product(std::vector<float> u, std::vector<float> v) {
        if (u.size() != v.size()) {
            std::cerr << "Vectors u and v are not of the same length." << std::endl;
        }
        
        float result = 0.0;
        
        for (size_t i = 0; i < u.size(); i++) {
            result += u[i] * v[i];
        }

        return result;
    }
    
    /*
    Activation functions
    */
    float sigmoid(float x) {
        return 1.0f / (1.0f + std::exp(-x));
    }

    float sigmoid_prime(float x) {
        return sigmoid(x) * (1.0f - sigmoid(x));
    }

    float relu(float x) {
        return std::max(0.0f, x);
    }

    float relu_prime(float x) {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    float leaky_relu(float x) {
        return std::max(0.1f * x, x);
    }

    float leaky_relu_prime(float x) {
        return x > 0.0f ? 1.0f : 0.01f;
    }

    std::vector<float> softmax(const std::vector<float> &input) {
        std::vector<float> out;
        float exp_sum = 0.0f;

        for (size_t i = 0; i < input.size(); ++i) {
            exp_sum += std::exp(input[i]);
        }

        for (size_t j = 0; j < input.size(); ++j) {
            out.push_back(std::exp(input[j]) / exp_sum);
        }
        return out;
    }

    std::vector<int> one_hot_label(int label) {
        std::vector<int> one_hot_result;

        for (int i = 0; i < 10; ++i) {
            i == label ? one_hot_result.push_back(1) : one_hot_result.push_back(0);
        }

        return one_hot_result;
    }

    /*
      Loss function
    */
    float cross_entropy_loss(const std::vector<float> &output, const std::vector<int> &one_hot_label) {
        float loss = 0.0f;
        float epsilon = 1e-8;  // prevent log(0)

        for (size_t i = 0; i < output.size(); ++i) {
            float p = std::max(epsilon, output[i]);
            loss += one_hot_label[i] * std::log(p);
        }

        return -loss;
    }
}
