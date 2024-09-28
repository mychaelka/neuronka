#include <iostream>
#include <vector>
#include "math.h"

int main() {

    // dot product
    std::vector<int> u = {1, 2, 3};
    std::vector<int> v = {1, 2, 3};
    std::vector<int> w = {2, 3, 4, 5, 6};

    int dot = dot_product(u,v);
    std::cout << dot << std::endl;

    int bad_dot = dot_product(v, w);
}