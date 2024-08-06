#include <iostream>
#include "box_constraints.h"
#include <memory>

int main() {
    constexpr int state_dim = 6;
    constexpr int control_dim = 2;

    Eigen::Matrix<double, state_dim, 1> state_min;
    Eigen::Matrix<double, state_dim, 1> state_max;
    Eigen::Matrix<double, control_dim, 1> control_min;
    Eigen::Matrix<double, control_dim, 1> control_max;

    state_min << -10, -10, -M_PI, -M_PI / 3, -5, -1;
    state_max << 10, 10, M_PI, M_PI / 3, 5, 1; 
    control_min << -1, -1;
    control_max << 1, 1;

    BoxConstraints<state_dim, control_dim> box_constraints(state_min, state_max, control_min, control_max);
    
    // Constraints<state_dim, control_dim, 2 * (state_dim + control_dim)>* ptr = &box_constraints;

    Eigen::Matrix<double, state_dim, 1> x;
    Eigen::Matrix<double, control_dim, 1> u;

    x << 1.5, -1.5, 0, 0, 0, 0;
    u << 0.51, -0.51;

    auto c = box_constraints.constraints(x, u);
    std::cout << "Constraints: \n" << c << std::endl;

    auto box_jacobian = box_constraints.constraints_jacobian(x, u);
    std::cout << "Jacobian A: \n" << box_jacobian.first << std::endl;
    std::cout << "Jacobian B: \n" << box_jacobian.second << std::endl;

    auto box_aug_jacobian = box_constraints.augmented_lagrangian_jacobian(x, u);
    std::cout << "box aug Jacobian A: \n" << box_aug_jacobian.first << std::endl;
    std::cout << "box aug Jacobian B: \n" << box_aug_jacobian.second << std::endl;

    auto box_hessian = box_constraints.constraints_hessian(x, u);
    std::cout << "Hessian hx: \n" << std::get<0>(box_hessian)[0] << std::endl;
    std::cout << "Hessian hu: \n" << std::get<1>(box_hessian)[0] << std::endl;
    std::cout << "Hessian hxu: \n" << std::get<2>(box_hessian)[0] << std::endl;

    auto box_aug_hessian = box_constraints.augmented_lagrangian_hessian(x, u);
    std::cout << "Aug Hessian hx: \n" << std::get<0>(box_aug_hessian) << std::endl;
    std::cout << "Aug Hessian hu: \n" << std::get<1>(box_aug_hessian) << std::endl;
    std::cout << "Aug Hessian hxu: \n" << std::get<2>(box_aug_hessian) << std::endl;



    return 0;
}
