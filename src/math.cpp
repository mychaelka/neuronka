#include "math.h"
#include <vector>
#include <stdexcept>

int dot_product(std::vector<int> u, std::vector<int> v) {
    int result = 0;

    if (u.size() != v.size()) {
        throw std::length_error("Vectors u and v are not of the same length.");
    }

    for (size_t i = 0; i < u.size(); i++) {
        result += u[i] * v[i];
    }

    return result;
}