#include <iostream>
#include <chrono>
#include <thread>
#include <array>
#include <vector>
#include <random>
#include "model/new_bicycle_node.h"
#include "constraints/box_constraints.h"
#include "new_al_ilqr.h"



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
    double L = 3.0;
    int num_points = 50;
    std::vector<Eigen::VectorXd> goal_list_fast = generateSShapeGoalFull(v, dt, num_points);

    Eigen::MatrixXd Q_fast = Eigen::MatrixXd::Zero(6, 6);
    Q_fast.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6;
    Q_fast *= 1e3;
    Eigen::MatrixXd R_fast = Eigen::MatrixXd::Identity(2, 2) * 1e2;


    std::array<Eigen::Matrix<double, 6, 1>, 2> state_bounds;
    state_bounds[0] << -1000, -1000, -2 * M_PI, -10, -100, -10;
    state_bounds[1] << 1000, 1000, 2 * M_PI, 10, 100, 10;

    std::array<Eigen::Matrix<double, 2, 1>, 2> control_bounds;
    control_bounds[0] << -0.2, -1;
    control_bounds[1] << 0.2, 1;

    BoxConstraints<6, 2> constraints_obj(state_bounds[0], state_bounds[1], control_bounds[0], control_bounds[1]);

    std::vector<std::shared_ptr<NewILQRNode<6, 2>>> ilqr_node_list;

    ilqr_node_list.clear();

    for (int i = 0; i <= num_points; ++i) {
        ilqr_node_list.push_back(std::make_shared<NewBicycleNode<BoxConstraints<6, 2>>>(L, dt, 0.001, goal_list_fast[i], Q_fast, R_fast, constraints_obj));
    }

    Eigen::Matrix<double, 6,1> init_state;

    init_state << 0.0, 0.0, 0.0, 0.0, v, 0.0;

    NewALILQR<6,2> solver(ilqr_node_list, init_state);

    solver.linearizedInitialGuess();
    solver.CalcDerivatives();
    solver.Backward();
    auto forward_start = std::chrono::high_resolution_clock::now();
    solver.Forward();
    auto forward_end = std::chrono::high_resolution_clock::now();

    auto x_list = solver.get_x_list();
    auto u_list = solver.get_u_list();
    auto K = solver.get_K();
    auto k = solver.get_k();
    double alpha = 1.0;

    
    auto multi_start_jacobian = std::chrono::high_resolution_clock::now();
    auto jacobian = ParallelFullBicycleDynamicJacobianRK2<50>(x_list.block(0,0,6,50), u_list.block(0,0,2,50), 0.1, 3.0, 0.001);
    auto multi_end_jacobian = std::chrono::high_resolution_clock::now();

    auto jacobian_raw = ilqr_node_list[0]->dynamics_jacobian(x_list.block(0,0,6,1), u_list.block(0,0,2,1));
    auto signal_jacobian = std::chrono::high_resolution_clock::now();

    std::cout << "x " << x_list.block(0,0,6,1) << "\n " <<"u " << u_list.block(0,0,2,1) << std::endl;

    for(int index = 0; index < 1; ++index) {
        std::cout << jacobian_raw.first << "\n" << jacobian.first.block(0,index * 6,6,6) << "\n" << std::endl;
    }

    
    double best_alpha = 1.0;
    double best_cost = 0.0;
    auto multi_start = std::chrono::high_resolution_clock::now();
    solver.ParallelLinearSearch(alpha, best_alpha, best_cost);
    auto multi_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> multi_count = std::chrono::duration<double>(multi_end - multi_start);

    std::chrono::duration<double> forward_count = std::chrono::duration<double>(forward_end - forward_start);

    std::chrono::duration<double> multi_jacobian = std::chrono::duration<double>(multi_end_jacobian - multi_start_jacobian);

    std::chrono::duration<double> signal_jacobian_c = std::chrono::duration<double>(signal_jacobian - multi_end_jacobian);

    std::cout << "multi linear search" << multi_count.count() << " seconds" << std::endl;
    std::cout << "forward " << forward_count.count() << " seconds" << std::endl;

    std::cout << "multi_jacobian " << multi_jacobian.count() << " seconds" << std::endl;
    std::cout << "signal_jacobian_c " << signal_jacobian_c.count() << " seconds" << std::endl;
    

    return 0;
}
