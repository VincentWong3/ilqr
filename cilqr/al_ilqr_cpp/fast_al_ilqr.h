#ifndef FASTALILQR_H
#define FASTALILQR_H

#include <array>
#include <Eigen/Dense>
#include <tuple>
#include "model/fast_ilqr_node.h"

template<int state_dim, int control_dim, int horizon>
class FastALILQR {
public:
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;
    using VectorConstraint = Eigen::Matrix<double, 2 * (control_dim + state_dim), 1>;
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;
    using MatrixQ = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixR = Eigen::Matrix<double, control_dim, control_dim>;
    using MatrixK = Eigen::Matrix<double, control_dim, state_dim>;

    FastALILQR(const std::array<FastILQRNode<state_dim, control_dim>*, horizon + 1>& ilqr_nodes);
    std::pair<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> linearizedInitialGuess();
    double computeTotalCost();
    std::pair<Eigen::Matrix<double, control_dim, horizon>, Eigen::Matrix<double, control_dim, state_dim * horizon>> backward();
    std::pair<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> forward(const Eigen::Matrix<double, control_dim, horizon>& k, const Eigen::Matrix<double, control_dim, state_dim * horizon>& K);
    std::tuple<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>, Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> optimize(int max_iters = 30, double tol = 1e-4);
    void updateLambda();
    void updateMu(double gain);
    double computeConstraintViolation();

private:
    std::array<FastILQRNode<state_dim, control_dim>*, horizon + 1> ilqr_nodes_;
    int horizon_;
    double regulation_gain_ = 0.0;
    double pre_violation_ = 0.0;
    double violation_ = 0.0;
    double inner_loop_final_cost_ = 0.0;
    double dV_ = 0.0;
    bool need_regulation_ = false;
    double regulation_increase_factor_ = 1.F;
    int inner_loop_count_ = 0;
    int outer_loop_count_ = 0;
    MatrixR Quu_reg_;

    // 临时变量
    Eigen::Matrix<double, state_dim, horizon + 1> x_;
    Eigen::Matrix<double, control_dim, horizon> u_;
    Eigen::Matrix<double, state_dim, horizon + 1> new_x_;
    Eigen::Matrix<double, control_dim, horizon> new_u_;
    Eigen::Matrix<double, control_dim, state_dim * horizon> K_;
    Eigen::Matrix<double, control_dim, horizon> k_;
    Eigen::Matrix<double, control_dim, horizon> pre_k_;


    // Jacobi and Hessian containers
    std::array<MatrixA, horizon> A_;
    std::array<MatrixB, horizon> B_;
    std::array<VectorState, horizon + 1> cost_Jx_;
    std::array<VectorControl, horizon> cost_Ju_;
    std::array<MatrixQ, horizon + 1> cost_Hx_;
    std::array<MatrixR, horizon> cost_Hu_;
};

template<int state_dim, int control_dim, int horizon>
FastALILQR<state_dim, control_dim, horizon>::FastALILQR(const std::array<FastILQRNode<state_dim, control_dim>*, horizon + 1>& ilqr_nodes)
    : ilqr_nodes_(ilqr_nodes), horizon_(horizon) {}

template<int state_dim, int control_dim, int horizon>
std::pair<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> FastALILQR<state_dim, control_dim, horizon>::linearizedInitialGuess() {
    
    // MatrixQ P = ilqr_nodes_[horizon_]->costHessian().first;
    MatrixQ P = ilqr_nodes_[horizon_]->costHessian().first.Identity();
    Quu_reg_ = ilqr_nodes_[horizon_]->costHessian().second;

    std::array<MatrixK, horizon> K_list;
    x_.col(0) = ilqr_nodes_[0]->state();

    for (int t = horizon_ - 1; t >= 0; --t) {
        FastILQRNode<state_dim, control_dim>* current_node = ilqr_nodes_[t];
        auto dynamics_jacobian = current_node->dynamicsJacobian(current_node->goal(), VectorControl::Zero());
        MatrixA A = dynamics_jacobian.first;
        MatrixB B = dynamics_jacobian.second;

        MatrixK K = (current_node->costHessian().second.Identity() * 20.0 + B.transpose() * P * B).inverse() * (B.transpose() * P * A);
        K_list[t] = K;
        P = current_node->costHessian().first.Identity() + A.transpose() * P * (A - B * K);

    }

    for (int t = 0; t < horizon_; ++t) {
        FastILQRNode<state_dim, control_dim>* current_node = ilqr_nodes_[t];
        VectorState goal_state = current_node->goal();
        MatrixK K = K_list[t];

        u_.col(t) = -K * (x_.col(t) - goal_state);
        u_.col(t) = u_.col(t).cwiseMin(current_node->control_max()).cwiseMax(current_node->control_min());
        x_.col(t + 1).noalias() = current_node->dynamics(x_.col(t), u_.col(t));

        current_node->setControl(u_.col(t));
        ilqr_nodes_[t + 1]->setState(x_.col(t + 1));
    }

    for (auto* node : ilqr_nodes_) {
        node->setLambda(VectorConstraint::Zero(node->constraint_dim));
        node->setMu(1.0);
    }

    return {x_, u_};
}

template<int state_dim, int control_dim, int horizon>
double FastALILQR<state_dim, control_dim, horizon>::computeTotalCost() {
    double total_cost = 0;
    for (auto* node : ilqr_nodes_) {
        total_cost += node->cost();
    }
    return total_cost;
}

template<int state_dim, int control_dim, int horizon>
std::pair<Eigen::Matrix<double, control_dim, horizon>, Eigen::Matrix<double, control_dim, state_dim * horizon>> FastALILQR<state_dim, control_dim, horizon>::backward() {
    for (int t = 0; t < horizon_; ++t) {
        FastILQRNode<state_dim, control_dim>* node = ilqr_nodes_[t];
        auto dynamics_jacobian = node->dynamicsJacobian(node->state(), node->control());
        A_[t] = dynamics_jacobian.first;
        B_[t] = dynamics_jacobian.second;
        auto cost_jacobian = node->costJacobian();
        cost_Jx_[t] = cost_jacobian.first;
        cost_Ju_[t] = cost_jacobian.second;
        auto cost_hessian = node->costHessian();
        cost_Hx_[t] = cost_hessian.first;
        cost_Hu_[t] = cost_hessian.second;
    }

    auto cost_jacobian = ilqr_nodes_[horizon_]->costJacobian();
    cost_Jx_[horizon_] = cost_jacobian.first;
    auto cost_hessian = ilqr_nodes_[horizon_]->costHessian();
    cost_Hx_[horizon_] = cost_hessian.first;

    VectorState Vx = cost_Jx_[horizon_];
    MatrixQ Vxx = cost_Hx_[horizon_];

    dV_ = 0;

    for (int t = horizon_ - 1; t >= 0; --t) {
        VectorControl Qu = cost_Ju_[t] + B_[t].transpose() * Vx;
        VectorState Qx = cost_Jx_[t] + A_[t].transpose() * Vx;
        Eigen::Matrix<double, control_dim, state_dim> Qux = B_[t].transpose() * Vxx * A_[t];
        MatrixR Quu = cost_Hu_[t] + B_[t].transpose() * Vxx * B_[t];
        MatrixQ Qxx = cost_Hx_[t] + A_[t].transpose() * Vxx * A_[t];
        MatrixR Quu_inv;
        if (need_regulation_) {
            Quu_inv = (Quu + Quu_reg_ * 0.0).inverse();
        } else {
            Quu_inv = Quu.inverse();
        }


        K_.block(0, t * state_dim, control_dim, state_dim).noalias() = -Quu_inv * Qux;
        k_.col(t).noalias() = -Quu_inv * Qu;

        Vx.noalias() = Qx + Qux.transpose() * k_.col(t);
        Vxx.noalias() = Qxx + Qux.transpose() * K_.block(0, t * state_dim, control_dim, state_dim);

        dV_ += (k_.col(t).transpose() * Qu)(0,0);
    }
    std::cout << "dv" << dV_ << std::endl;
    return {k_, K_};
}

template<int state_dim, int control_dim, int horizon>
std::pair<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> FastALILQR<state_dim, control_dim, horizon>::forward(const Eigen::Matrix<double, control_dim, horizon>& k, const Eigen::Matrix<double, control_dim, state_dim * horizon>& K) {
    double alpha = 1.0;

    for (int i = 0; i <= horizon_; ++i) {
        x_.col(i) = ilqr_nodes_[i]->state();
    }

    for (int i = 0; i < horizon_; ++i) {
        u_.col(i) = ilqr_nodes_[i]->control();
    }

    new_x_.col(0) = x_.col(0);
    double old_cost = computeTotalCost();

    while (alpha > 1e-6) {
        for (int t = 0; t < horizon_; ++t) {
            new_u_.col(t) = u_.col(t) + alpha * k.col(t) + K.block(0, t * state_dim, control_dim, state_dim) * (new_x_.col(t) - x_.col(t));
            new_x_.col(t + 1).noalias() = ilqr_nodes_[t]->dynamics(new_x_.col(t), new_u_.col(t));
            ilqr_nodes_[t + 1]->setState(new_x_.col(t + 1));
            ilqr_nodes_[t]->setControl(new_u_.col(t));
        }
        double new_cost = computeTotalCost();
            
        if (new_cost < old_cost) {
            break;
        } else {
            if (outer_loop_count_ <= 1) {
                double alpha_inv = 1.0 / alpha;
                double temp = (new_cost - old_cost) * alpha_inv * alpha_inv - dV_ * alpha_inv;
                alpha = - dV_ / (2 * temp);
            } else { 
                alpha = alpha / 2;
            }
        }
    }
    if (alpha <= 1e-6) {
        new_x_ = x_;
        new_u_ = u_;
        for (int t = 0; t < horizon_; ++t) {
            ilqr_nodes_[t + 1]->setState(new_x_.col(t + 1));
            ilqr_nodes_[t]->setControl(new_u_.col(t));
        }
        std::cout << "linear_search failed" << std::endl;
    }

    return {new_x_, new_u_};
}

template<int state_dim, int control_dim, int horizon>
std::tuple<Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>, Eigen::Matrix<double, state_dim, horizon + 1>, Eigen::Matrix<double, control_dim, horizon>> FastALILQR<state_dim, control_dim, horizon>::optimize(int max_iters, double tol) {
    auto start_total = std::chrono::high_resolution_clock::now();

    auto linearized_initial_guess = linearizedInitialGuess();
    Eigen::Matrix<double, state_dim, horizon + 1> x_init = linearized_initial_guess.first;
    Eigen::Matrix<double, control_dim, horizon> u_init = linearized_initial_guess.second;
    double old_cost = computeTotalCost();

    Eigen::Matrix<double, state_dim, horizon + 1> x = x_init;
    Eigen::Matrix<double, control_dim, horizon> u = u_init;

    Eigen::Matrix<double, control_dim, horizon> previous_k = Eigen::Matrix<double, control_dim, horizon>::Zero();

    for (int j = 0; j < 20; ++j) {
        outer_loop_count_ = j;
        old_cost = computeTotalCost();
        need_regulation_ = true;
        auto start_inner = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 20; ++i) {
            inner_loop_count_ = i;
            auto backward_result = backward();

            Eigen::Matrix<double, control_dim, horizon> k = backward_result.first;
            Eigen::Matrix<double, control_dim, state_dim * horizon> K = backward_result.second;

            Eigen::Matrix<double, control_dim, horizon> diff = (k - previous_k).cwiseAbs();
            double max_diff = diff.maxCoeff();
            
            if (max_diff < 1e-4) {
                std::cout << "Converged with max_diff: " << max_diff << std::endl;
                break;
            }
            previous_k = k;


            auto forward_result = forward(k, K);

            Eigen::Matrix<double, state_dim, horizon + 1> new_x = forward_result.first;
            Eigen::Matrix<double, control_dim, horizon> new_u = forward_result.second;
            double new_cost = computeTotalCost();
            std::cout << "inner loop count " << i << " steps" << std::endl;
            std::cout << "old_cost " << old_cost << std::endl; 
            std::cout << "new_cost " << new_cost << std::endl; 

            if ((std::abs(new_cost - old_cost) < tol)) {
                inner_loop_final_cost_ = new_cost;
                break;   
            }
            x = new_x;
            u = new_u;
            old_cost = new_cost;
        }
        auto end_inner = std::chrono::high_resolution_clock::now();

        violation_ = computeConstraintViolation();
        std::chrono::duration<double> total_inner = end_inner - start_inner;


        std::cout << "inner loop took " << total_inner.count() << " seconds." << std::endl;
        std::cout << "outer loop count " << j << " steps" << std::endl;
        std::cout << "\n" << std::endl;
        std::cout << "violation \n" << violation_  << std::endl;

        if (violation_ < 1e-3) {
            if (j == 1) {
              updateLambda();
              std::cout << "lambda is updated ++++++++" << std::endl;
            } else {
              break;
            }
        } else if (violation_ <= 1e-1 && violation_ > 1e-3) {
            updateLambda();
            std::cout << "lambda is updated ++++++++" << std::endl;

            if (std::fabs(pre_violation_ - violation_) < 1e-4) {
                break;
            }
        } else {
            if (j == 0) {
              updateMu(8);
            } else {
              if (j >= 20) {
                updateLambda();
                std::cout << "lambda is updated ++++++++" << std::endl;
              } else {
                updateMu(8);
              }
            }
        }
        pre_violation_ = violation_;
    }

    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_elapsed = end_total - start_total;
    std::cout << "Total optimization took " << total_elapsed.count() << " seconds." << std::endl;

    return std::make_tuple(x_init, u_init, x, u);
}

template<int state_dim, int control_dim, int horizon>
void FastALILQR<state_dim, control_dim, horizon>::updateLambda() {
    for (auto* node : ilqr_nodes_) {
        node->updateLambda();
    }
}

template<int state_dim, int control_dim, int horizon>
void FastALILQR<state_dim, control_dim, horizon>::updateMu(double gain) {
    for (auto* node : ilqr_nodes_) {
        node->setMu(node->mu() * gain);
    }
}

template<int state_dim, int control_dim, int horizon>
double FastALILQR<state_dim, control_dim, horizon>::computeConstraintViolation() {
    double violation = 0.0;
    for (auto* node : ilqr_nodes_) {
        Eigen::VectorXd one_constrain = node->constraints().cwiseMax(-node->lambda() / node->mu());
        // std::cout << "one constrain  " << one_constrain << std::endl;
        double one_violation = one_constrain.transpose() * one_constrain;
        violation += std::sqrt(one_violation);
    }
    return violation;
}

#endif // FASTALILQR_H
