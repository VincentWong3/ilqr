#include <iostream>
#include <vector>
#include <array>
#include <Eigen/Dense>
#include <chrono>
#include "fast_al_ilqr.h"
#include "model/fast_full_bicycle_dynamic_node.h"

// Function to generate S-shape goal
std::vector<Eigen::VectorXd> generateSShapeGoalFull(double v, double dt, int num_points) {
    std::vector<Eigen::VectorXd> goals;
    for (int i = 0; i <= num_points; ++i) {
        double t = i * dt;
        double x = v * t;
        double y = 50 * std::sin(0.1 * t);
        double theta = std::atan2(50 * 0.1 * std::cos(0.1 * t), v);
        double dx = v;
        double dy = 50 * 0.1 * std::cos(0.1 * t);
        double ddx = 0;
        double ddy = -50 * 0.1 * 0.1 * std::sin(0.1 * t);
        double curvature = (dx * ddy - dy * ddx) / std::pow(dx * dx + dy * dy, 1.5);
        double delta = std::atan(curvature * 1.0);
        Eigen::VectorXd goal_state(6);
        goal_state << x, y, theta, delta, v, 0;  // (x, y, theta, delta, v_desire, a_desire)
        goals.push_back(goal_state);
    }
    return goals;
}

int main() {
    double v = 10;
    double dt = 0.1;
    double L = 1;
    int num_points = 50;
    std::vector<Eigen::VectorXd> goal_list_fast = generateSShapeGoalFull(v, dt, num_points);

    Eigen::MatrixXd Q_fast = Eigen::MatrixXd::Zero(6, 6);
    Q_fast.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6;
    Q_fast *= 1e3;
    Eigen::MatrixXd R_fast = Eigen::MatrixXd::Identity(2, 2) * 1e2;

    std::array<Eigen::Matrix<double, 6, 1>, 2> state_bounds_fast;
    state_bounds_fast[0] << -1000, -1000, -2 * M_PI, -10, -100, -10;
    state_bounds_fast[1] << 1000, 1000, 2 * M_PI, 10, 100, 10;

    std::array<Eigen::Matrix<double, 2, 1>, 2> control_bounds_fast;
    control_bounds_fast[0] << -0.2, -1;
    control_bounds_fast[1] << 0.2, 1;

    std::array<FastILQRNode<6, 2>*, 51> fast_ilqr_nodes;
    for (int i = 0; i <= num_points; ++i) {
        fast_ilqr_nodes[i] = new FastFullBicycleDynamicNode(L, dt, 0.0001, state_bounds_fast, control_bounds_fast, goal_list_fast[i], Q_fast, R_fast);
    }

    FastALILQR<6, 2, 50> fast_ilqr(fast_ilqr_nodes);

    fast_ilqr_nodes[0]->setState(Eigen::Matrix<double, 6, 1>::Zero());
    fast_ilqr_nodes[0]->setState(4, v);  // 设置初始速度

    auto start_total = std::chrono::high_resolution_clock::now();
    auto result = fast_ilqr.optimize();
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = end_total - start_total;
    std::cout << "Total optimization took " << elapsed_total.count() << " seconds." << std::endl;

    auto x_init_fast = std::get<0>(result);
    auto u_init_fast = std::get<1>(result);
    auto x_opt_fast = std::get<2>(result);
    auto u_opt_fast = std::get<3>(result);

    std::cout << "Initial State Trajectory:\n";
    for (int i = 0; i < x_init_fast.cols(); ++i) {
        std::cout << "Step " << i << ": " << x_init_fast.col(i).transpose() << "\n";
    }

    std::cout << "\nOptimized Control Inputs:\n";
    for (int i = 0; i < u_opt_fast.cols(); ++i) {
        std::cout << "Step " << i << ": " << u_opt_fast.col(i).transpose() << "\n";
    }

    for (auto node : fast_ilqr_nodes) {
        delete node;
    }

    return 0;
}
