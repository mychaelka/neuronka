#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#pragma once

/* Load data vectors (train or test) */
static std::vector<std::vector<float>> parse_input(const std::string &vectors_file) {
    
    std::vector<std::vector<float>> out_data;
    std::ifstream file(vectors_file);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file " << vectors_file << std::endl;
        return out_data;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<float> row;
        std::stringstream line_stream(line);
        std::string cell;
        
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell));
        }
        
        out_data.push_back(row);
    }

    file.close();
    return out_data;
}


/* Load label vectors (train or test) */
static std::vector<float> parse_labels(const std::string &labels_file) {
    std::vector<float> out_labels;
    std::ifstream file(labels_file);

    if (!file.is_open()) {
        std::cerr << "Error: Could not open input file." << labels_file << std::endl;
        return out_labels;
    }

    std::string line;
    while (std::getline(file, line)) {
        out_labels.push_back(std::stof(line));
    }

    file.close();
    return out_labels;
}

/* Write predictions to file */
void write_predictions(const std::string& filename, const std::vector<float>& predictions) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + filename);
    }

    for (const auto& prediction : predictions) {
        file << static_cast<int>(prediction) << "\n";
    }

    file.close();
}
