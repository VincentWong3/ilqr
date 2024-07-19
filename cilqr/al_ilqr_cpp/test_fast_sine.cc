#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <random>
#include "fast_sine.h"

// 周期
#define TWO_PI 6.283185307179586476925286766559
#define PI 3.141592653589793238462643383279
#define HALF_PI 1.5707963267948966192313216916398

int main() {
    const int num_points = 1;
    std::vector<double> angles(num_points);
    std::vector<double> fast_sin_results(num_points);
    std::vector<double> fast_cos_results(num_points);
    std::vector<double> std_sin_results(num_points);
    std::vector<double> std_cos_results(num_points);

    // 随机生成 (-2π, 2π) 范围内的角度
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-100 * TWO_PI, 100 * TWO_PI);

    for (int i = 0; i < num_points; ++i) {
        angles[i] = dis(gen);
    }
    double a = 0.0;
    double b = 0.0;

    // 计算 fast_sin 和 fast_cos 的值并测量时间
    auto start_fast = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_points; ++i) {
        a += fast_sin(angles[i]);
        b += fast_cos(angles[i]);
    }
    auto end_fast = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> fast_duration = end_fast - start_fast;

    // 计算 std::sin 和 std::cos 的值并测量时间
    auto start_std = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_points; ++i) {
        a -= std::sin(angles[i]);
        b -= std::cos(angles[i]);
    }
    auto end_std = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> std_duration = end_std - start_std;

    std::cout << "Fast sin/cos calculation time: " << fast_duration.count() << " seconds\n";
    std::cout << "Std sin/cos calculation time: " << std_duration.count() << " seconds\n";
    std::cout << "sin calculation error: " << a / num_points << " seconds\n";
    std::cout << "cos calculation error: " << b / num_points << " seconds\n";



    return 0;
}
