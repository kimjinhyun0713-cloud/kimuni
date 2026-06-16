#include <iostream>
#include <vector>
#include "function.hpp"

int main() {
    std::vector<std::vector<double>> bounds = {
        {0.0, 10.0},
        {0.0, 10.0},
        {0.0, 10.0}
    };
    auto [lattice, angle, zeropoint] = calLattice(bounds);
    
    std::cout << "Lattice a: " << lattice[0] << "\n";
    std::cout << "Angle alpha: " << angle[0] << "\n";
    
    return 0;
}