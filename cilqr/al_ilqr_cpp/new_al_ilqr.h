#ifndef NEW_ALILQR_H
#define NEW_ALILQR_H

#include <array>
#include <Eigen/Dense>
#include <tuple>
#include "model/new_bicycle_node.h"
#include "constraints/box_constraints.h"
#include <iostream>
#include <chrono>


template<int state_dim, int control_dim, int horizon>
class NewALILQR {
public:
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;
    using VectorConstraint = Eigen::Matrix<double, 2 * (control_dim + state_dim), 1>;
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;
    using MatrixQ = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixR = Eigen::Matrix<double, control_dim, control_dim>;
    using MatrixK = Eigen::Matrix<double, control_dim, state_dim>;

    NewALILQR(const std::array<NewILQRNode<state_dim, control_dim>*, horizon + 1>& ilqr_nodes, const VectorState& init_state, int real_horizon)
        : ilqr_nodes_(ilqr_nodes), init_state_(init_state), real_horizon_(real_horizon) { zero_control_.setZero();
        zero_state_.setZero(); }
    void linearizedInitialGuess();
    void CalcDerivatives(int start, int end);
    void CalcDerivatives();
    double computeTotalCost();

    void UpdateTrajectoryAndCostList(double alpha);

    void ParallelLinearSearch(double alpha, double& best_alpha, double& best_cost);

    Eigen::Matrix<double, state_dim, PARALLEL_NUM> State_Dot(const Eigen::Matrix<double, state_dim, PARALLEL_NUM>& x,
                                                             const Eigen::Matrix<double, control_dim, PARALLEL_NUM>& u);
    

    void Backward();
    void Forward();
    
    void UpdateMu(double gain);

    void UpdateLambda();

    double ComputeConstraintViolation();
    void optimize(int max_outer_iter, int max_inner_iter, double max_violation);

    void ILQRProcess(int max_iter, double max_tol);

    Eigen::Matrix<double, state_dim, horizon + 1> get_x_list() { return x_list_; }
    Eigen::Matrix<double, control_dim, horizon> get_u_list() { return u_list_; }
    std::array<MatrixK, horizon> get_K() {return K_list_; }
    std::array<VectorControl, horizon> get_k() {return k_list_; }
    std::array<Eigen::Matrix<double, state_dim, state_dim>, horizon> get_jacobian_x() { return dynamics_jacobian_x_list_; }
    std::array<Eigen::Matrix<double, state_dim, control_dim>, horizon> get_jacobian_u() { return dynamics_jacobian_u_list_; }




private:
    Eigen::Matrix<double, state_dim, horizon + 1> x_list_;
    Eigen::Matrix<double, control_dim, horizon> u_list_;
    Eigen::Matrix<double, state_dim, horizon + 1> pre_x_list_;
    Eigen::Matrix<double, control_dim, horizon> pre_u_list_;
    Eigen::Matrix<double, state_dim, horizon + 1> x_temp_list_;
    Eigen::Matrix<double, control_dim, horizon> u_temp_list_;

    Eigen::Matrix<double, control_dim, 1> zero_control_;
    Eigen::Matrix<double, state_dim, 1> zero_state_;

    std::array<Eigen::Matrix<double, state_dim, 1>, horizon + 1> cost_augmented_lagrangian_jacobian_x_list_;
    std::array<Eigen::Matrix<double, control_dim, 1>, horizon> cost_augmented_lagrangian_jacobian_u_list_;
    std::array<Eigen::Matrix<double, state_dim, state_dim>, horizon + 1> cost_augmented_lagrangian_hessian_x_list_;
    std::array<Eigen::Matrix<double, control_dim, control_dim>, horizon> cost_augmented_lagrangian_hessian_u_list_;
    std::array<Eigen::Matrix<double, state_dim, state_dim>, horizon> dynamics_jacobian_x_list_;
    std::array<Eigen::Matrix<double, state_dim, control_dim>, horizon> dynamics_jacobian_u_list_;

    Eigen::Matrix<double, horizon + 1, 1> max_constraints_violation_list_;

    std::array<MatrixK, horizon> K_list_;
    std::array<VectorControl, horizon> k_list_;

    double deltaV_linear_ = 0.F;
    double deltaV_quadratic_ = 0.F;
    double mu_ = 1.0;
    Eigen::Matrix<double, horizon + 1, 1> cost_list_;
    std::array<std::tuple<MatrixA, MatrixA, MatrixA>, horizon> dynamics_hession_x_list_;

public:
std::array<NewILQRNode<state_dim, control_dim>*, horizon + 1> ilqr_nodes_;
VectorState init_state_;
int real_horizon_ = 10;

};

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::linearizedInitialGuess() {
    x_list_.col(0) = init_state_;
    u_list_.col(0).setZero();
    MatrixQ P = ilqr_nodes_[horizon]->cost_hessian(zero_state_, zero_control_).first.Identity();
    for (int t = horizon - 1; t >= 0; --t) {
        auto dynamics_jacobian = ilqr_nodes_[t]->dynamics_jacobian(ilqr_nodes_[t]->goal(), VectorControl::Zero());
        MatrixA A = dynamics_jacobian.first;
        MatrixB B = dynamics_jacobian.second;

        MatrixK K = (ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).second.Identity() * 20.0 + B.transpose() * P * B).inverse() * (B.transpose() * P * A);
        K_list_[t] = K;
        P = ilqr_nodes_[t]->cost_hessian(zero_state_, zero_control_).first.Identity() + A.transpose() * P * (A - B * K);
    }

    for (int t = 0; t < horizon; ++t) {
        VectorState goal_state = ilqr_nodes_[t]->goal();
        MatrixK K = K_list_[t];

        u_list_.col(t) = -K * (x_list_.col(t) - goal_state);
        x_list_.col(t + 1) = ilqr_nodes_[t]->dynamics(x_list_.col(t), u_list_.col(t));

    }
    for (auto* node : ilqr_nodes_) {
        node->reset_lambda();
        node->reset_mu();
    }
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::CalcDerivatives(int start, int end) {
    // double cost_aug_hessian_time_sum = 0.0;
    // double cost_aug_jacobian_time_sum = 0.0;

    for (int index = start; index <= end; ++index) {
        auto x = x_list_.col(index);
        auto u = u_list_.col(index);
        cost_list_[index] = ilqr_nodes_[index]->cost(x, u);
        //auto start_cost_jacobian = std::chrono::high_resolution_clock::now();
        auto cost_augmented_lagrangian_jacobian = ilqr_nodes_[index]->cost_jacobian(x, u);
        //auto start_cost_hessian = std::chrono::high_resolution_clock::now();
        auto cost_augmented_lagrangian_hessian = ilqr_nodes_[index]->cost_hessian(x, u);
        //auto end_cost_hessian = std::chrono::high_resolution_clock::now();
        auto dynamics_jacobian = ilqr_nodes_[index]->dynamics_jacobian(x, u);
        cost_augmented_lagrangian_jacobian_x_list_[index] = cost_augmented_lagrangian_jacobian.first;
        cost_augmented_lagrangian_jacobian_u_list_[index] = cost_augmented_lagrangian_jacobian.second;
        cost_augmented_lagrangian_hessian_x_list_[index] = cost_augmented_lagrangian_hessian.first;
        cost_augmented_lagrangian_hessian_u_list_[index] = cost_augmented_lagrangian_hessian.second;
        dynamics_jacobian_x_list_[index] = dynamics_jacobian.first;
        dynamics_jacobian_u_list_[index] = dynamics_jacobian.second;
        dynamics_hession_x_list_[index] = ilqr_nodes_[index]->dynamics_hessian_fxx(x, u);
        
        // std::chrono::duration<double> cost_aug_jacobian_dur = std::chrono::duration_cast<std::chrono::duration<double>>(start_cost_hessian - start_cost_jacobian);
        // std::chrono::duration<double> cost_aug_hessian_dur = std::chrono::duration_cast<std::chrono::duration<double>>(end_cost_hessian - start_cost_hessian);
        // cost_aug_jacobian_time_sum += cost_aug_jacobian_dur.count();
        // cost_aug_hessian_time_sum +=  cost_aug_hessian_dur.count();
    }
    // std::cout << "cost_aug_jacobian_time_sum " << cost_aug_jacobian_time_sum << std::endl;
    // std::cout << "cost_aug_hessian_time_sum " << cost_aug_hessian_time_sum << std::endl;

    // for (int index = start; index <= end; ++index) {
    //     auto x = x_list_.col(index);
    //     auto u = u_list_.col(index);
    //     ilqr_nodes_[index]->CalcAllCost(x, u, cost_list_[index],
    //                                     cost_augmented_lagrangian_jacobian_x_list_[index],
    //                                     cost_augmented_lagrangian_jacobian_u_list_[index],
    //                                     cost_augmented_lagrangian_hessian_x_list_[index],
    //                                     cost_augmented_lagrangian_hessian_u_list_[index]);
    //     auto dynamics_jacobian = ilqr_nodes_[index]->dynamics_jacobian(x, u);
    //     dynamics_jacobian_x_list_[index] = dynamics_jacobian.first;
    //     dynamics_jacobian_u_list_[index] = dynamics_jacobian.second;
    //     dynamics_hession_x_list_[index] = ilqr_nodes_[index]->dynamics_hessian_fxx(x, u);
    // }
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::CalcDerivatives() {
    auto x_end = x_list_.col(horizon);
    cost_list_[horizon] = ilqr_nodes_[horizon]->cost(x_end, zero_control_);
    cost_augmented_lagrangian_jacobian_x_list_[horizon] = ilqr_nodes_[horizon]->cost_jacobian(x_end, zero_control_).first;
    cost_augmented_lagrangian_hessian_x_list_[horizon] = ilqr_nodes_[horizon]->cost_hessian(x_end, zero_control_).first;
    
    // auto start = std::chrono::high_resolution_clock::now();
    CalcDerivatives(0, horizon - 1);
    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> CalcDerivatives_duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    // std::cout << "calc der " << CalcDerivatives_duration.count() << "seconds" << std::endl;
}

template<int state_dim, int control_dim, int horizon>
double NewALILQR<state_dim, control_dim, horizon>::computeTotalCost() {
    return cost_list_.sum();
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::UpdateTrajectoryAndCostList(double alpha) {
    for(int i = 0; i < horizon; ++i) {
        u_list_.col(i) += K_list_[i] * (x_list_.col(i) - pre_x_list_.col(i)) + alpha * k_list_[i];
        x_list_.col(i + 1) = ilqr_nodes_[i]->dynamics(x_list_.col(i), u_list_.col(i));
        cost_list_[i] = ilqr_nodes_[i]->cost(x_list_.col(i), u_list_.col(i));
    }
    cost_list_[horizon] = ilqr_nodes_[horizon]->cost(x_list_.col(horizon), zero_control_);
}

template<int state_dim, int control_dim, int horizon>
Eigen::Matrix<double, state_dim, PARALLEL_NUM> NewALILQR<state_dim, control_dim, horizon>::State_Dot(const Eigen::Matrix<double, state_dim, PARALLEL_NUM>& x,
                                                             const Eigen::Matrix<double, control_dim, PARALLEL_NUM>& u) {

    auto theta_list_matrix_raw = x.row(2);
    auto delta_list_matrix_raw = x.row(3);
    auto v_list_matrix_raw = x.row(4);
    auto a_list_matrix_raw = x.row(5);
    auto v_cos_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().cos();
    auto v_sin_theta_array = v_list_matrix_raw.array() * theta_list_matrix_raw.array().sin();
    auto v_tan_delta_array_divide_L = v_list_matrix_raw.array() * delta_list_matrix_raw.array().tan() * (v_list_matrix_raw.Ones().array() + 0.001 * v_list_matrix_raw.array() * v_list_matrix_raw.array()).inverse() / 3.0;

    Eigen::Matrix<double, 6, PARALLEL_NUM> answer;
    answer.row(0) = v_cos_theta_array.matrix();
    answer.row(1) = v_sin_theta_array.matrix();
    answer.row(2) = v_tan_delta_array_divide_L.matrix();
    answer.row(3) = u.row(0);
    answer.row(4) = a_list_matrix_raw;
    answer.row(5) = u.row(1);
    return answer;
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::ParallelLinearSearch(double alpha, double& best_alpha, double& best_cost) {
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    auto x_list_raw = x_list_;
    auto k_forward = k_list_;

    Eigen::Matrix<double, PARALLEL_NUM, 1> alpha_vector;

    for (int index = 0; index < PARALLEL_NUM; ++index) {
        alpha_vector[index] = alpha;
        alpha /= 2.0;
    }
    Eigen::Array<double, 2, PARALLEL_NUM> real_alpha = (alpha_vector.transpose().replicate(control_dim, 1)).array();
    
    Eigen::Matrix<double, PARALLEL_NUM, PARALLEL_NUM> alpha_matrix = alpha_vector.asDiagonal();

    

    Eigen::Matrix<double, state_dim, PARALLEL_NUM> x_old = x_list_raw.col(0).replicate(1, PARALLEL_NUM);

    Eigen::Matrix<double, state_dim, PARALLEL_NUM> x_new = x_old;

    Eigen::Matrix<double, control_dim, PARALLEL_NUM> k_one;

    Eigen::Matrix<double, control_dim, PARALLEL_NUM> u_old;

    Eigen::Matrix<double, control_dim, PARALLEL_NUM> u_new;

    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost_list_;

    parallel_cost_list_.setZero();

    Eigen::Matrix<double, PARALLEL_NUM, 1> one_cost_list;

    auto loop_start = std::chrono::high_resolution_clock::now();

    Eigen::Matrix<double, 10, 1>  loop_count;
    loop_count.setZero();


    for (int index = 0; index < horizon; index++) {
        x_old = x_list_raw.col(index).replicate(1, PARALLEL_NUM);
        u_old = u_list_.col(index).replicate(1, PARALLEL_NUM);
        k_one = k_forward[index].replicate(1, PARALLEL_NUM);
        k_one = (k_one.array() * real_alpha).matrix();
        u_new = u_old + K_list_[index] * (x_new - x_old) + k_one;
        one_cost_list = ilqr_nodes_[index]->parallel_cost(x_new, u_new);
        parallel_cost_list_ += one_cost_list;
        auto x_dot = State_Dot(x_new, u_new);
        auto x_mid = x_new + x_dot * 0.05;
        auto x_dot_mid = State_Dot(x_mid, u_new);
        x_new = x_new + x_dot_mid * 0.1;
    }
    one_cost_list = ilqr_nodes_[horizon]->parallel_cost(x_new, zero_control_.replicate(1, PARALLEL_NUM));
    parallel_cost_list_ += one_cost_list;
    Eigen::Index min_index;
    best_cost = parallel_cost_list_.minCoeff(&min_index);
    int real_index = static_cast<int>(min_index);
    best_alpha = alpha_matrix(real_index, real_index);
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::Backward() {
    auto Vx = cost_augmented_lagrangian_jacobian_x_list_[horizon];
    auto Vxx = cost_augmented_lagrangian_hessian_x_list_[horizon];
    deltaV_linear_ = 0.0;
    deltaV_quadratic_ = 0.0;

    for (int t = horizon - 1; t >= 0; --t) {
        auto A = dynamics_jacobian_x_list_[t];
        auto B = dynamics_jacobian_u_list_[t];

        VectorControl Qu = cost_augmented_lagrangian_jacobian_u_list_[t] + B.transpose() * Vx;
        VectorState Qx = cost_augmented_lagrangian_jacobian_x_list_[t] + A.transpose() * Vx;
        Eigen::Matrix<double, control_dim, state_dim> Qux = B.transpose() * Vxx * A;
        MatrixR Quu = cost_augmented_lagrangian_hessian_u_list_[t] + B.transpose() * Vxx * B;
        MatrixQ Qxx = cost_augmented_lagrangian_hessian_x_list_[t] + A.transpose() * Vxx * A;
        Qxx += std::get<0>(dynamics_hession_x_list_[t]) * Vx[0] + std::get<1>(dynamics_hession_x_list_[t]) * Vx[1] + std::get<2>(dynamics_hession_x_list_[t]) * Vx[2];
        MatrixR Quu_inv;
        Quu_inv = (Quu).inverse();
        // auto info = Quu_chol.compute(Quu_inv).info();
        // if (info != Eigen::Success) {
        //     Vx = cost_Jx_[horizon_];
        //     Vxx = cost_Hx_[horizon_];
        //     IncreaseRegGain();
        //     break;
        // }
        auto K = -Quu_inv * Qux;
        auto k = -Quu_inv * Qu;
        K_list_[t] = K;
        k_list_[t] = k;

        Vx.noalias() = Qx + K.transpose() * (Quu * k + Qu) + Qux.transpose() * k;
        Vxx.noalias() = Qxx + K.transpose() * (Quu * K + Qux) + Qux.transpose() * K;


        deltaV_linear_ += (k.transpose() * Qu)(0,0);
        deltaV_quadratic_ += 0.5 * (k.transpose() * Quu * k)(0,0);
    }
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::Forward() {
    double old_cost = computeTotalCost();
    double new_cost = 0.0;
    pre_x_list_ = x_list_;
    pre_u_list_ = u_list_;
    auto pre_cost_list_ = cost_list_;
    double alpha = 1.0;
    double best_alpha = 1.0;
    double best_cost = 0.0;

    // auto para_start = std::chrono::high_resolution_clock::now();
    // auto para_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> CalcDerivatives_duration = std::chrono::duration_cast<std::chrono::duration<double>>(para_end - para_start);
    // std::cout << "parallel " << CalcDerivatives_duration.count() << "seconds" << std::endl;
    // std::cout << "best_alpha " << best_alpha << std::endl;
    // auto para_start = std::chrono::high_resolution_clock::now;
    
    // auto para_end = std::chrono::high_resolution_clock::now;
    // std::chrono::duration<double> para_duration = std::chrono::duration_cast<std::chrono::duration<double>>(para_end - para_start);
    // std::cout << "parallel duration " << para_duration.count() << std::endl;


    if (std::fabs(deltaV_linear_) < 0.2F) {
        return;
    }

    for (int index = 0; index < 10; ++index) {
        UpdateTrajectoryAndCostList(alpha);
        new_cost = computeTotalCost();
        if (new_cost < old_cost) {
            break;
        }
        alpha /= 2.0;
        x_list_ = pre_x_list_;
        u_list_ = pre_u_list_;
    }
    if (new_cost >= old_cost) {
        ParallelLinearSearch(alpha, best_alpha, best_cost);
        if (best_cost >= old_cost) {
            x_list_ = pre_x_list_;
            u_list_ = pre_u_list_;
            cost_list_ = pre_cost_list_;
        } else {
          UpdateTrajectoryAndCostList(best_alpha);
          new_cost = best_cost;
        }
    }
}

template<int state_dim, int control_dim, int horizon>
double NewALILQR<state_dim, control_dim, horizon>::ComputeConstraintViolation() {
    for (int index = 0; index < horizon; ++index) {
        max_constraints_violation_list_[index] = ilqr_nodes_[index]->max_constraints_violation(x_list_.col(index), u_list_.col(index));
    }
    max_constraints_violation_list_[horizon] = ilqr_nodes_[horizon]->max_constraints_violation(x_list_.col(horizon), zero_control_);
    return max_constraints_violation_list_.maxCoeff();
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::ILQRProcess(int max_iter, double max_tol) {
    using namespace std::chrono;

    for(int iter = 0; iter < max_iter; ++iter) {
        //auto start_CalcDerivatives = high_resolution_clock::now();
        CalcDerivatives();
        //auto end_CalcDerivatives = high_resolution_clock::now();
        //duration<double> CalcDerivatives_duration = duration_cast<duration<double>>(end_CalcDerivatives - start_CalcDerivatives);
        //std::cout << "CalcDerivatives took " << CalcDerivatives_duration.count() << " seconds" << std::endl;

        double old_cost = cost_list_.sum();

        //auto start_Backward = high_resolution_clock::now();
        Backward();
        //auto end_Backward = high_resolution_clock::now();
        //<double> Backward_duration = duration_cast<duration<double>>(end_Backward - start_Backward);
        //std::cout << "Backward took " << Backward_duration.count() << " seconds" << std::endl;

        // auto start_Forward = high_resolution_clock::now();
        Forward();
        // auto end_Forward = high_resolution_clock::now();
        // duration<double> Forward_duration = duration_cast<duration<double>>(end_Forward - start_Forward);
        // std::cout << "Forward took " << Forward_duration.count() << " seconds" << std::endl;

        double new_cost = cost_list_.sum();

        if ((old_cost - new_cost < max_tol) && ((old_cost - new_cost) >= 0)) {
            break;
        }
    }
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::UpdateMu(double gain) {
   mu_ = mu_ * gain;
   for(auto* node : ilqr_nodes_) {
      node->update_mu(mu_);
   }
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::UpdateLambda() {
   for(int index = 0; index < horizon; ++index) {
      ilqr_nodes_[index]->update_lambda(x_list_.col(index), u_list_.col(index));
   }
   ilqr_nodes_[horizon]->update_lambda(x_list_.col(horizon), zero_control_);
}

template<int state_dim, int control_dim, int horizon>
void NewALILQR<state_dim, control_dim, horizon>::optimize(int max_outer_iter, int max_inner_iter, double max_violation) {
    using namespace std::chrono;
    auto start_optimize = high_resolution_clock::now();

    linearizedInitialGuess();
    for (int index = 0; index < max_outer_iter; ++index) {
        // auto start_ILQR = high_resolution_clock::now();
        ILQRProcess(max_inner_iter, 1e-3);
        // auto end_ILQR = high_resolution_clock::now();
        // duration<double> ILQR_duration = duration_cast<duration<double>>(end_ILQR - start_ILQR);
        // std::cout << "ILQRProcess took " << ILQR_duration.count() << " seconds" << std::endl;

        double inner_violation = ComputeConstraintViolation();
        // std::cout << "inner_violation" << inner_violation << std::endl;
        if (inner_violation < max_violation) {
            break;
        } else {
            if (inner_violation > 5 * max_violation) {
                UpdateMu(100.0);
            } else {
                UpdateLambda();
            }
        } 
    }
    auto end_optimize = high_resolution_clock::now();
    duration<double> optimize_duration = duration_cast<duration<double>>(end_optimize - start_optimize);

    for(int i = 0; i < horizon - 1; ++i) {
        std::cout << "u_result " << u_list_.col(i).transpose() << std::endl;
    }
    
    std::cout << "optimize took " << optimize_duration.count() << " seconds" << std::endl;
}

#endif // NEW_ALILQR_H
