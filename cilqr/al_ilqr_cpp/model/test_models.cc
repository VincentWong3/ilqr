#include <iostream>
#include <Eigen/Dense>
#include <array>
#include <vector>
#include <chrono>
#include <random>
#include "fast_full_bicycle_dynamic_node.h"
#include "full_bicycle_dynamic_node.h"

using namespace std::chrono;

int main() {
    const int num_tests = 10000;

    // 初始化随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-10, 10);
    std::uniform_real_distribution<> angle_dist(-M_PI, M_PI);
    std::uniform_real_distribution<> steering_angle_dist(-M_PI / 3, M_PI / 3);
    std::uniform_real_distribution<> control_dist(-1, 1);

    // 示例目标状态和Q、R矩阵
    Eigen::Matrix<double, 6, 1> goal;
    goal << 0, 0, 0, 0, 0, 0;
    Eigen::Matrix<double, 6, 6> Q = Eigen::Matrix<double, 6, 6>::Identity();
    Eigen::Matrix<double, 2, 2> R = Eigen::Matrix<double, 2, 2>::Identity();

    // 定义状态和控制的边界
    std::array<Eigen::Matrix<double, 6, 1>, 2> state_bounds;
    state_bounds[0] << -10, -10, -M_PI, -M_PI / 3, -5, -1; // 最小值
    state_bounds[1] << 10, 10, M_PI, M_PI / 3, 5, 1;       // 最大值

    std::array<Eigen::Matrix<double, 2, 1>, 2> control_bounds;
    control_bounds[0] << -1, -1; // 最小值
    control_bounds[1] << 1, 1;   // 最大值

    std::vector<Eigen::VectorXd> state_bounds_vec = {state_bounds[0], state_bounds[1]};
    std::vector<Eigen::VectorXd> control_bounds_vec = {control_bounds[0], control_bounds[1]};

    // 创建节点对象
    FastFullBicycleDynamicNode fast_dynamic_node(2.0, 0.1, 0.0001, state_bounds, control_bounds, goal, Q, R);
    FullBicycleDynamicNode full_dynamic_node(2.0, 0.1, 0.0001, state_bounds_vec, control_bounds_vec, goal, Q, R);

    // 计时变量
    double total_duration_fast = 0;
    double total_duration_full = 0;
    double max_diff_dynamics = 0;
    double max_diff_dynamicsJacobian = 0;
    double max_diff_costJacobian = 0;
    double max_diff_costHessian = 0;
    double max_diff_constraintJacobian = 0;
    double max_diff_cost = 0;
    double max_diff_constraints = 0;

    // 测试函数列表
    std::vector<std::pair<const char*, std::function<void()>>> test_functions = {
        {"dynamics", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_dynamics = 0;
            for (int i = 0; i < num_tests; ++i) {
                Eigen::Matrix<double, 6, 1> test_state;
                Eigen::Matrix<double, 2, 1> test_control;
                for (int j = 0; j < 2; ++j) test_state[j] = dist(gen);
                test_state[2] = angle_dist(gen);
                test_state[3] = steering_angle_dist(gen);
                for (int j = 4; j < 6; ++j) test_state[j] = dist(gen);
                for (int j = 0; j < 2; ++j) test_control[j] = control_dist(gen);

                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.dynamics(test_state, test_control);
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.dynamics(test_state, test_control);
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast - result_full).norm();
                if (diff > max_diff_dynamics) max_diff_dynamics = diff;
            }

            std::cout << "dynamics:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_dynamics << "\n\n";
        }},
        {"dynamicsJacobian", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_dynamicsJacobian = 0;
            for (int i = 0; i < num_tests; ++i) {
                Eigen::Matrix<double, 6, 1> test_state;
                Eigen::Matrix<double, 2, 1> test_control;
                for (int j = 0; j < 2; ++j) test_state[j] = dist(gen);
                test_state[2] = angle_dist(gen);
                test_state[3] = steering_angle_dist(gen);
                for (int j = 4; j < 6; ++j) test_state[j] = dist(gen);
                for (int j = 0; j < 2; ++j) test_control[j] = control_dist(gen);

                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.dynamicsJacobian(test_state, test_control);
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.dynamicsJacobian(test_state, test_control);
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast.first - result_full.first).norm() + (result_fast.second - result_full.second).norm();
                if (diff > max_diff_dynamicsJacobian) max_diff_dynamicsJacobian = diff;
            }

            std::cout << "dynamicsJacobian:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_dynamicsJacobian << "\n\n";
        }},
        {"costJacobian", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_costJacobian = 0;
            for (int i = 0; i < num_tests; ++i) {
                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.costJacobian();
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.costJacobian();
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast.first - result_full.first).norm() + (result_fast.second - result_full.second).norm();
                if (diff > max_diff_costJacobian) max_diff_costJacobian = diff;
            }

            std::cout << "costJacobian:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_costJacobian << "\n\n";
        }},
        {"costHessian", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_costHessian = 0;
            for (int i = 0; i < num_tests; ++i) {
                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.costHessian();
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.costHessian();
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast.first - result_full.first).norm() + (result_fast.second - result_full.second).norm();
                if (diff > max_diff_costHessian) max_diff_costHessian = diff;
            }

            std::cout << "costHessian:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_costHessian << "\n\n";
        }},
        {"constraintJacobian", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_constraintJacobian = 0;
            for (int i = 0; i < num_tests; ++i) {
                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.constraintJacobian();
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.constraintJacobian();
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast.first - result_full.first).norm() + (result_fast.second - result_full.second).norm();
                if (diff > max_diff_constraintJacobian) max_diff_constraintJacobian = diff;
            }

            std::cout << "constraintJacobian:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_constraintJacobian << "\n\n";
        }},
        {"cost", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_cost = 0;
            for (int i = 0; i < num_tests; ++i) {
                auto start = high_resolution_clock::now();
                double result_fast = fast_dynamic_node.cost();
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                double result_full = full_dynamic_node.cost();
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = std::abs(result_fast - result_full);
                if (diff > max_diff_cost) max_diff_cost = diff;
            }

            std::cout << "cost:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_cost << "\n\n";
        }},
        {"constraints", [&]() {
            total_duration_fast = 0;
            total_duration_full = 0;
            max_diff_constraints = 0;
            for (int i = 0; i < num_tests; ++i) {
                auto start = high_resolution_clock::now();
                auto result_fast = fast_dynamic_node.constraints();
                auto end = high_resolution_clock::now();
                total_duration_fast += duration_cast<microseconds>(end - start).count();

                start = high_resolution_clock::now();
                auto result_full = full_dynamic_node.constraints();
                end = high_resolution_clock::now();
                total_duration_full += duration_cast<microseconds>(end - start).count();

                double diff = (result_fast - result_full).norm();
                if (diff > max_diff_constraints) max_diff_constraints = diff;
            }

            std::cout << "constraints:\n";
            std::cout << "Fast average duration: " << total_duration_fast / num_tests << " µs, Full average duration: " << total_duration_full / num_tests << " µs\n";
            std::cout << "Max result difference: " << max_diff_constraints << "\n\n";
        }}
    };

    for (auto& test : test_functions) {
        std::cout << "Testing " << test.first << "...\n";
        test.second();
    }

    return 0;
}
