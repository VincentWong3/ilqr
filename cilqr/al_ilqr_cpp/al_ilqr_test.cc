#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "al_ilqr.h"

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
    std::vector<Eigen::VectorXd> goal_list_full = generateSShapeGoalFull(v, dt, num_points);

    Eigen::MatrixXd Q_full = Eigen::MatrixXd::Zero(6, 6);
    Q_full.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6;
    Eigen::MatrixXd R_full = Eigen::MatrixXd::Identity(2, 2) * 10;

    std::vector<Eigen::VectorXd> state_bounds_full(2);
    state_bounds_full[0] = Eigen::VectorXd(6);
    state_bounds_full[0] << -1000, -1000, -2 * M_PI, -10, -100, -10;
    state_bounds_full[1] = Eigen::VectorXd(6);
    state_bounds_full[1] << 1000, 1000, 2 * M_PI, 10, 100, 10;

    std::vector<Eigen::VectorXd> control_bounds_full(2);
    control_bounds_full[0] = Eigen::VectorXd(2);
    control_bounds_full[0] << -0.2, -1;
    control_bounds_full[1] = Eigen::VectorXd(2);
    control_bounds_full[1] << 0.2, 1;

    std::vector<ILQRNode*> ilqr_nodes_full;
    for (const auto& goal : goal_list_full) {
        ilqr_nodes_full.push_back(new FullBicycleDynamicNode(L, dt, 0.0001, state_bounds_full, control_bounds_full, goal, Q_full, R_full));
    }

    ALILQR ilqr_full(ilqr_nodes_full);

    ilqr_nodes_full[0]->setState(Eigen::VectorXd::Zero(6));
    ilqr_nodes_full[0]->setState(4, v);  // 设置初始速度

    auto result = ilqr_full.optimize();
    Eigen::MatrixXd x_init_full = std::get<0>(result);
    Eigen::MatrixXd u_init_full = std::get<1>(result);
    Eigen::MatrixXd x_opt_full = std::get<2>(result);
    Eigen::MatrixXd u_opt_full = std::get<3>(result);

    // 输出初始和优化后的状态轨迹
    std::cout << "Initial State Trajectory:\n";
    for (int i = 0; i < x_init_full.cols(); ++i) {
        std::cout << "Step " << i << ": " << x_init_full.col(i).transpose() << "\n";
    }

    std::cout << "\nOptimized State Trajectory:\n";
    for (int i = 0; i < x_opt_full.cols(); ++i) {
        std::cout << "Step " << i << ": " << x_opt_full.col(i).transpose() << "\n";
    }

    // 输出优化后的控制量
    std::cout << "\nOptimized Control Inputs:\n";
    for (int i = 0; i < u_opt_full.cols(); ++i) {
        std::cout << "Step " << i << ": " << u_opt_full.col(i).transpose() << "\n";
    }

    // 清理内存
    for (ILQRNode* node : ilqr_nodes_full) {
        delete node;
    }

    return 0;
}
