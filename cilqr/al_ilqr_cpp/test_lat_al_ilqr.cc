#include "constraints/box_constraints.h"
#include "constraints/quadratic_constraints.h"
#include "new_al_ilqr.h"
#include "model/new_lat_bicycle_node.h"
#include <memory>
#include <vector>
#include <array>

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
        Eigen::VectorXd goal_state(4);
        goal_state << x, y, theta, delta;  // (x, y, theta, delta, v_desire, a_desire)
        goals.push_back(goal_state);
    }
    return goals;
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> GenerateCycleEquations(double centre_x, double centre_y, double r, int x_dims) {
    Eigen::MatrixXd Q(x_dims, x_dims);
    Eigen::MatrixXd A(1, x_dims);
    Eigen::MatrixXd C(1, 1);
    C << r * r - centre_x * centre_x - centre_y * centre_y;
    Q.setZero();
    A.setZero();
    Q(0, 0) = -1.0;
    Q(1, 1) = -1.0;
    A(0, 0) = 2 * centre_x;
    A(0, 1) = 2 * centre_y;
    return {Q, A, C};
}


int main() {
    double v = 10;
    double dt = 0.1;
    double L = 3.0;
    int num_points = 50;
    std::vector<Eigen::VectorXd> goal_list_fast = generateSShapeGoalFull(v, dt, num_points);

    Eigen::MatrixXd Q_fast = Eigen::MatrixXd::Zero(4, 4);
    Q_fast.diagonal() << 1e-1, 1e-1, 1e-0, 1e-9;
    Q_fast *= 1e3;
    Eigen::MatrixXd R_fast = Eigen::MatrixXd::Identity(1, 1) * 1e2;

    Eigen::Matrix<double, 10, 4> A;
    A.setZero();
    Eigen::Matrix<double, 10, 1> B;
    B.setZero();
    B(0, 0) = 1;
    B(1, 0) = -1;
    Eigen::Matrix<double, 10, 1> C;
    C.setZero();
    C(0, 0) = -0.2;
    C(1, 0) = -0.2;


    LinearConstraints<4, 1, 10> constraints_obj(A, B, C);

    constraints_obj.set_current_constraints_index(1);

    std::vector<std::shared_ptr<NewILQRNode<4, 1>>> ilqr_node_list;

    ilqr_node_list.clear();

    for (int i = 0; i <= num_points; ++i) {
        ilqr_node_list.push_back(std::make_shared<NewLatBicycleNode<LinearConstraints<4, 1, 10>>>(L, dt, 0.001, v, goal_list_fast[i], Q_fast, R_fast, constraints_obj));
    }

    Eigen::Matrix<double, 4,1> init_state;

    init_state << 0.0, 0.0, 0.0, 0.0;

    Eigen::Matrix<double, 2, 4> left_car;
    left_car << 32, 32, 28, 28,
                14, 16, 16, 14;
    std::vector<Eigen::Matrix<double, 2, 4>> left_obs;
    std::vector<Eigen::Matrix<double, 2, 4>> right_obs;
    left_obs.push_back(left_car);
    right_obs.clear();


    NewALILQR<4,1> solver(ilqr_node_list, init_state, right_obs, left_obs);

    solver.optimize(50, 100, 1e-3);

    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "u_result " << solver.get_u_list().col(i).transpose() << std::endl;
    }

    for(int i = 0; i < num_points - 1; ++i) {
        std::cout << "x_result " << solver.get_x_list().col(i).transpose() << std::endl;
    }


    // constexpr  int constraint_dim = 5;
    // Eigen::Matrix<double, constraint_dim, 6> A;
    // Eigen::Matrix<double, constraint_dim, 2> B;
    // Eigen::Matrix<double, constraint_dim, 1> C;
    // A.setZero();
    // B.setZero();
    // C.setZero();
    // B << 0, 0, 1, 0, 0, 1, -1, 0, 0, -1;
    // C << 0, -0.2, -1, -0.2, -1;
    // std::array<Eigen::Matrix<double, 6, 6>, constraint_dim> Q;

    // for (int i = 0; i < constraint_dim; ++i) {
    //     Q[i] = Eigen::Matrix<double, 6, 6>::Zero();
    // }

    // auto ans = GenerateCycleEquations(20.0, 12, 4.0, 6);

    // Q[0] = std::get<0>(ans);
    // C(0, 0) = (std::get<2>(ans)).value();
    // A.row(0) = std::get<1>(ans);

    // std::cout << 'A' << A << std::endl;

    // QuadraticConstraints<6, 2, constraint_dim> quad_constrants(Q, A, B, C);

    // std::vector<std::shared_ptr<NewILQRNode<6, 2>>> q_ilqr_node_list;

    // q_ilqr_node_list.clear();

    // for (int i = 0; i <= num_points; ++i) {
    //     q_ilqr_node_list.push_back(std::make_shared<NewBicycleNode<QuadraticConstraints<6, 2, constraint_dim>>>(L, dt, 0.001, goal_list_fast[i], Q_fast, R_fast, quad_constrants));
    // }

    // NewALILQR<6,2> q_solver(q_ilqr_node_list, init_state);

    // q_solver.optimize(30, 100, 1e-3);

    // for(int i = 0; i < num_points - 1; ++i) {
    //     std::cout << "u_result " << q_solver.get_u_list().col(i).transpose() << std::endl;
    // }

    // for(int i = 0; i < num_points - 1; ++i) {
    //     std::cout << "x_result " << q_solver.get_x_list().col(i).transpose() << std::endl;
    // }

    

    return 0;
}