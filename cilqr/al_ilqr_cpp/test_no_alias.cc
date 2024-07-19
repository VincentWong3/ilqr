#include <iostream>
#include <Eigen/Dense>
#include <chrono>

using namespace std::chrono;
using namespace Eigen;

void testNoAlias() {
    const int size = 1000;
    MatrixXd A = MatrixXd::Random(size, size);
    MatrixXd B = MatrixXd::Random(size, size);
    MatrixXd C(size, size);
    
    auto start = high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        C = A * B;
    }
    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "Without noalias: " << elapsed.count() << " seconds." << std::endl;
    
    start = high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        C.noalias() = A * B;
    }
    end = high_resolution_clock::now();
    elapsed = end - start;
    std::cout << "With noalias: " << elapsed.count() << " seconds." << std::endl;
}

int main() {
    testNoAlias();
    return 0;
}
