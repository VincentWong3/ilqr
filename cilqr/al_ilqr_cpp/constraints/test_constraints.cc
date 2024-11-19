#include <iostream>
#include "box_constraints.h"
#include "quadratic_constraints.h"
#include "dynamic_linear_constraints.h"
#include <memory>

int main() {
    constexpr int state_dim = 6;
    constexpr int control_dim = 2;
    constexpr int constraint_dim = 2;


    Eigen::Matrix<double, state_dim, 1> state_min;
    Eigen::Matrix<double, state_dim, 1> state_max;
    Eigen::Matrix<double, control_dim, 1> control_min;
    Eigen::Matrix<double, control_dim, 1> control_max;


    state_min << -1, -1, -1, -1, -1, -1;
    state_max << 1, 1, 1, 1, 1, 1;
    control_min << -2, -2;
    control_max << 2, 2;

    BoxConstraints<state_dim, control_dim> box_constraints(state_min, state_max, control_min, control_max);

    
    // Constraints<state_dim, control_dim, 2 * (state_dim + control_dim)>* ptr = &box_constraints;

    std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> Q;
    Eigen::Matrix<double, constraint_dim, state_dim> A;
    Eigen::Matrix<double, constraint_dim, control_dim> B;
    Eigen::Matrix<double, constraint_dim, 1> C;

    for (int i = 0; i < constraint_dim; ++i) {
        Q[i] = Eigen::Matrix<double, state_dim, state_dim>::Identity() * (i + 1);
    }
    A.setConstant(1.0);
    B.setConstant(0.0);
    C.setConstant(0.0);

    DynamicLinearConstraints<state_dim, control_dim> dynamic_linear_constraints(A, B, C);

    std::cout << "dynamic linear constraints" << std::endl;


    QuadraticConstraints<state_dim, control_dim, constraint_dim> quad_constraints(Q, A, B, C);

    Eigen::Matrix<double, state_dim, 1> x;
    Eigen::Matrix<double, control_dim, 1> u;

    x << 1, 1, 1, 1, 1, 1;
    u << 1, 1;

    Eigen::Matrix<double, state_dim, 2> x_para;
    Eigen::Matrix<double, control_dim, 2> u_para;

    x_para << 1,2,1,2,1,2,1,2,1,2,1,2;
    u_para << 1,2,1,2;


    auto c = box_constraints.constraints(x, u);
    std::cout << "Constraints: \n" << c << std::endl;

    auto c_dynamic = dynamic_linear_constraints.constraints(x, u);

    std::cout << "Dynamic Constraints: \n" << c_dynamic << std::endl;



    auto box_jacobian = box_constraints.constraints_jacobian(x, u);
    std::cout << "Jacobian A: \n" << box_jacobian.first << std::endl;
    std::cout << "Jacobian B: \n" << box_jacobian.second << std::endl;

    auto box_jacobian_dynamic = dynamic_linear_constraints.constraints_jacobian(x, u);
    std::cout << "dynamic Jacobian A: \n" << box_jacobian_dynamic.first << std::endl;
    std::cout << "dynamic Jacobian B: \n" << box_jacobian_dynamic.second << std::endl;

    double cost = box_constraints.augmented_lagrangian_cost(x, u);

    auto box_aug_jacobian = box_constraints.augmented_lagrangian_jacobian(x, u);
    std::cout << "box aug Jacobian A: \n" << box_aug_jacobian.first << std::endl;
    std::cout << "box aug Jacobian B: \n" << box_aug_jacobian.second << std::endl;

    double cost2 = dynamic_linear_constraints.augmented_lagrangian_cost(x, u);


    auto box_aug_jacobian_dynamic = dynamic_linear_constraints.augmented_lagrangian_jacobian(x, u);
    std::cout << "dynamic box aug Jacobian A: \n" << box_aug_jacobian_dynamic.first << std::endl;
    std::cout << "dynamic box aug Jacobian B: \n" << box_aug_jacobian_dynamic.second << std::endl;

    auto box_hessian = box_constraints.constraints_hessian(x, u);
    std::cout << "Hessian hx: \n" << std::get<0>(box_hessian)[0] << std::endl;
    std::cout << "Hessian hu: \n" << std::get<1>(box_hessian)[0] << std::endl;
    std::cout << "Hessian hxu: \n" << std::get<2>(box_hessian)[0] << std::endl;

    auto box_aug_hessian = box_constraints.augmented_lagrangian_hessian(x, u);
    std::cout << "Aug Hessian hx: \n" << std::get<0>(box_aug_hessian) << std::endl;
    std::cout << "Aug Hessian hu: \n" << std::get<1>(box_aug_hessian) << std::endl;
    std::cout << "Aug Hessian hxu: \n" << std::get<2>(box_aug_hessian) << std::endl;

    auto c_quad = quad_constraints.constraints(x, u);
    std::cout << "Quadratic Constraints: \n" << c_quad << std::endl;

    auto quad_jacobian = quad_constraints.constraints_jacobian(x, u);
    std::cout << "Quadratic Jacobian A: \n" << quad_jacobian.first << std::endl;
    std::cout << "Quadratic Jacobian B: \n" << quad_jacobian.second << std::endl;

    auto quad_hessian = quad_constraints.constraints_hessian(x, u);
    std::cout << "Quadratic Hessian hx: \n" << std::get<0>(quad_hessian)[0] << std::endl;
    std::cout << "Quadratic Hessian hu: \n" << std::get<1>(quad_hessian)[0] << std::endl;
    std::cout << "Quadratic Hessian hxu: \n" << std::get<2>(quad_hessian)[0] << std::endl;

    // auto para_quad_c = quad_constraints.parallel_constraints(x_para, u_para);

    // std::cout << "Quadratic para quad c: \n" << para_quad_c << std::endl;



    return 0;
}
