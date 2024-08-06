#ifndef PARALEL_COMPUTION_FUNCTION_H
#define PARALEL_COMPUTION_FUNCTION_H

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <cmath>

static bool is_parallel_initialized = false;
static Eigen::MatrixXd parallel_matrix_1, parallel_matrix_2, parallel_matrix_3, parallel_matrix_4;
static Eigen::MatrixXd ju_continuous_list;
static Eigen::MatrixXd Identify_list;

template <int parallel_num> void GenerateParallelFullBicycleJacobianMatrixBase() {
    parallel_matrix_1 = Eigen::Matrix<double, parallel_num, 6 * parallel_num>::Zero();
    parallel_matrix_2 = Eigen::Matrix<double, parallel_num, 6 * parallel_num>::Zero();
    parallel_matrix_3 = Eigen::Matrix<double, parallel_num, 6 * parallel_num>::Zero();
    parallel_matrix_4 = Eigen::Matrix<double, 1, 6 * parallel_num>::Zero();
    ju_continuous_list = Eigen::Matrix<double, 6, 2 * parallel_num>::Zero();
    Eigen::Matrix<double, 6, 2> ju_signal;
    Eigen::Matrix<double, 6, 6> identify = Eigen::Matrix<double, 6, 6>::Identity();
    Identify_list = identify.replicate(1, parallel_num);
    ju_signal.setZero();
    ju_signal(3, 0) = 1.0;
    ju_signal(5, 1) = 1.0;
    ju_continuous_list = ju_signal.replicate(1, parallel_num);

    for(int i = 0; i < parallel_num; ++i) {
        parallel_matrix_1(i, i * 6 + 2) = 1;
        parallel_matrix_2(i, i * 6 + 4) = 1;
        parallel_matrix_3(i, i * 6 + 3) = 1;
        parallel_matrix_4(0, i * 6 + 5) = 1;
    }
    is_parallel_initialized = true;
}




template <int parallel_num>
static Eigen::Matrix<double, 6, parallel_num> ParallelFullBicycleDynamicContinuous(const Eigen::Matrix<double, 6, parallel_num>& x,
                                                                                  const Eigen::Matrix<double, 2, parallel_num>& u,
                                                                                  double L, double k) {
    auto theta_list_matrix_raw = x.row(2);
    auto delta_list_matrix_raw = x.row(3);
    auto v_list_matrix_raw = x.row(4);
    auto a_list_matrix_raw = x.row(5);
    auto v_cos_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().cos();
    auto v_sin_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().sin();
    auto v_tan_delta_array_divide_L = v_list_matrix_raw.array() * delta_list_matrix_raw.array().tan() * ((v_list_matrix_raw.Ones().array() + k * v_list_matrix_raw.array() * v_list_matrix_raw.array()).inverse()) / L;

    Eigen::Matrix<double, 6, parallel_num> answer;
    answer.row(0) = v_cos_theta_array.matrix();
    answer.row(1) = v_sin_theta_array.matrix();
    answer.row(2) = v_tan_delta_array_divide_L.matrix();
    answer.row(3) = u.row(0);
    answer.row(4) = a_list_matrix_raw;
    answer.row(5) = u.row(1);
    return answer;
}

template <int parallel_num>
static Eigen::Matrix<double, 6, parallel_num> ParallelFullBicycleDynamicRK2(const Eigen::Matrix<double, 6, parallel_num>& x,
                                                                         const Eigen::Matrix<double, 2, parallel_num>& u,
                                                                         double dt, double L, double k) {
    auto x_dot = ParallelFullBicycleDynamicContinuous<parallel_num>(x, u, L, k);
    auto x_mid = x + x_dot * 0.5 * dt;
    auto x_dot_mid = ParallelFullBicycleDynamicContinuous<parallel_num>(x_mid, u, L, k);
    return x + dt * x_dot_mid;
}

template <int parallel_num>
static Eigen::Matrix<double, 6, parallel_num> ParallelFullBicycleDynamicRK4(const Eigen::Matrix<double, 6, parallel_num>& x,
                                                                            const Eigen::Matrix<double, 2, parallel_num>& u,
                                                                            double dt, double L, double k) {
    auto k1 = ParallelFullBicycleDynamicContinuous<parallel_num>(x, u, L, k);
    auto k2 = ParallelFullBicycleDynamicContinuous<parallel_num>(x + 0.5 * dt * k1, u, L, k);
    auto k3 = ParallelFullBicycleDynamicContinuous<parallel_num>(x + 0.5 * dt * k2, u, L, k);
    auto k4 = ParallelFullBicycleDynamicContinuous<parallel_num>(x + dt * k3, u, L, k);
    
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

template <int parallel_num>
static Eigen::Matrix<double, 6, parallel_num> ParallelJacobianElement(const Eigen::Matrix<double, 6, parallel_num>& x,
                                                                      const Eigen::Matrix<double, 2, parallel_num>& u,
                                                                      double dt, double L, double k) {
    Eigen::Matrix<double, 1, parallel_num> theta_list_matrix_raw = x.row(2);
    Eigen::Matrix<double, 1, parallel_num> delta_list_matrix_raw = x.row(3);
    Eigen::Matrix<double, 1, parallel_num> v_list_matrix_raw = x.row(4);
    auto v_list_array = v_list_matrix_raw.array();
    auto cos_theta_array = theta_list_matrix_raw.array().cos();
    auto sin_theta_array = theta_list_matrix_raw.array().sin();
    auto tan_delta_array = delta_list_matrix_raw.array().tan();
    auto sec_delta_array_square = delta_list_matrix_raw.Ones().array() + tan_delta_array * tan_delta_array;
    auto v_cos_theta_array = v_list_array * cos_theta_array;
    auto v_sin_theta_array = v_list_array * sin_theta_array;
    auto v_square_list = v_list_array * v_list_array;
    auto k_v_square_list = k * v_square_list;
    auto den = (L * (v_list_matrix_raw.Ones().array() + k_v_square_list)).inverse();
    auto den_square = den * den;
    auto k_v_square_minus_tan = tan_delta_array * (v_list_matrix_raw.Ones().array() - k_v_square_list) * L * den_square;
    auto v_sec_square_den = (v_list_array * sec_delta_array_square * den);
    Eigen::Matrix<double, 6, parallel_num> answer;
    answer.row(0) = -v_sin_theta_array.matrix();
    answer.row(1) = v_cos_theta_array.matrix();
    answer.row(2) = v_sec_square_den.matrix();
    answer.row(3) = cos_theta_array.matrix();
    answer.row(4) = sin_theta_array.matrix();
    answer.row(5) = k_v_square_minus_tan.matrix();
    return answer;
}

template <int parallel_num>
static std::pair<Eigen::Matrix<double, 6, 6 * parallel_num>, Eigen::Matrix<double, 6, 2 * parallel_num>> 
ParallelFullBicycleDynamicJacobianContinuous(const Eigen::Matrix<double, 6, parallel_num>& x,
                                   const Eigen::Matrix<double, 2, parallel_num>& u,
                                    double dt, double L, double k) {
    auto element = ParallelJacobianElement(x, u, dt, L, k);
    if (!is_parallel_initialized) {
        GenerateParallelFullBicycleJacobianMatrixBase<parallel_num>();
    }
    Eigen::Matrix<double, 2, parallel_num> v_cos_sin_list = element.block(0, 0, 2, parallel_num);
    


    Eigen::Matrix<double, 3, parallel_num> cos_sin_list = element.block(3, 0, 3, parallel_num);;

    Eigen::Matrix<double, 6, 6 * parallel_num> answer1;
    Eigen::Matrix<double, 6, 2 * parallel_num> answer2;


    answer1.setZero();
    answer1.block(0, 0, 2, 6 * parallel_num) += v_cos_sin_list * parallel_matrix_1;
    answer1.block(0, 0, 3, 6 * parallel_num) += cos_sin_list * parallel_matrix_2;
    answer1.block(2, 0, 1, 6 * parallel_num) += element.block(2, 0, 1, parallel_num) * parallel_matrix_3;
    answer1.row(4) = parallel_matrix_4;
    answer2 = ju_continuous_list;
    return {answer1, answer2};
}

template <int parallel_num>
static std::pair<Eigen::Matrix<double, 6, 6 * parallel_num>, Eigen::Matrix<double, 6, 2 * parallel_num>> 
ParallelFullBicycleDynamicJacobianRK2(const Eigen::Matrix<double, 6, parallel_num>& x,
                                   const Eigen::Matrix<double, 2, parallel_num>& u,
                                    double dt, double L, double k) {
    std::pair<Eigen::Matrix<double, 6, 6 * parallel_num>, Eigen::Matrix<double, 6, 2 * parallel_num>> 
    jacobian = ParallelFullBicycleDynamicJacobianContinuous<parallel_num>(x, u, dt, L, k);
    // auto x_dot = ParallelFullBicycleDynamicContinuous<parallel_num>(x, u, L, k);
    // auto x_mid = x + x_dot * 0.5 * dt;
    // std::pair<Eigen::Matrix<double, 6, 6 * parallel_num>, Eigen::Matrix<double, 6, 2 * parallel_num>>
    // jacobian_mid_continous = ParallelFullBicycleDynamicJacobianContinuous<parallel_num>(x_mid, u, dt, L, k);
    Eigen::Matrix<double, 6, 6 * parallel_num> jx = jacobian.first * dt + Identify_list;
    Eigen::Matrix<double, 6, 2 * parallel_num> ju = jacobian.second * dt;
    // Eigen::Matrix<double, 6, 6 * parallel_num> hessian_x = jacobian_mid_continous.first;
    // Eigen::Matrix<double, 6, 2 * parallel_num> hessian_u = jacobian_mid_continous.second;
    // for (int index = 0; index < parallel_num; ++index) {
    //     Eigen::Matrix<double, 6, 6> temp = (hessian_x.block(0, index * 6, 6, 6));
    //     hessian_x.block(0, index * 6, 6, 6) = temp * jacobian.first.block(0, index * 6, 6, 6) * dt * dt / 2.0;
    //     hessian_u.block(0, index * 2, 6, 2) = temp * jacobian.second.block(0, index * 2, 6, 2) * dt * dt / 2.0;
    // }
    // jx += hessian_x;
    // ju += hessian_u;
    return {jx, ju};
}

#endif // PARALEL_COMPUTION_FUNCTION_H

