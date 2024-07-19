#include "al_ilqr.h"
#include <iostream>
#include <chrono>

using namespace std::chrono;

ALILQR::ALILQR(const std::vector<ILQRNode*>& ilqr_nodes) : ilqr_nodes(ilqr_nodes), horizon(ilqr_nodes.size() - 1) {}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ALILQR::linearizedInitialGuess() {
    auto start = high_resolution_clock::now();

    int state_dim = ilqr_nodes[0]->state_dim();
    int control_dim = ilqr_nodes[0]->control_dim();
    
    Eigen::MatrixXd x = Eigen::MatrixXd::Zero(state_dim, horizon + 1);
    Eigen::MatrixXd u = Eigen::MatrixXd::Zero(control_dim, horizon);
    
    x.col(0) = ilqr_nodes[0]->state();
    
    Eigen::MatrixXd P = ilqr_nodes[horizon]->costHessian().first;
    std::vector<Eigen::MatrixXd> K_list;

    for (int t = horizon - 1; t >= 0; --t) {
        ILQRNode* current_node = ilqr_nodes[t];
        auto dynamics_jacobian = current_node->dynamicsJacobian(current_node->goal(), Eigen::VectorXd::Zero(control_dim));
        Eigen::MatrixXd A = dynamics_jacobian.first;
        Eigen::MatrixXd B = dynamics_jacobian.second;
        Eigen::MatrixXd K = (current_node->costHessian().second + B.transpose() * P * B).inverse() * (B.transpose() * P * A);
        K_list.push_back(K);
        P = current_node->costHessian().first + A.transpose() * P * (A - B * K);
    }

    std::reverse(K_list.begin(), K_list.end());

    for (int t = 0; t < horizon; ++t) {
        ILQRNode* current_node = ilqr_nodes[t];
        Eigen::VectorXd goal_state = current_node->goal();
        Eigen::MatrixXd K = K_list[t];

        u.col(t) = -K * (x.col(t) - goal_state);
        u.col(t) = u.col(t).cwiseMin(current_node->control_max()).cwiseMax(current_node->control_min());
        x.col(t + 1) = current_node->dynamics(x.col(t), u.col(t));

        current_node->setControl(u.col(t));
        ilqr_nodes[t + 1]->setState(x.col(t + 1));
    }

    for (ILQRNode* node : ilqr_nodes) {
        node->setLambda(Eigen::VectorXd::Zero(node->constraint_dim()));
        node->setMu(1.0);
    }

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;
    std::cout << "linearizedInitialGuess took " << elapsed.count() << " seconds." << std::endl;

    return {x, u};
}

double ALILQR::computeTotalCost() {
    
    double total_cost = 0;
    for (ILQRNode* node : ilqr_nodes) {
        total_cost += node->cost();
    }
    return total_cost;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ALILQR::backward() {
    
    int state_dim = ilqr_nodes[0]->state_dim();
    int control_dim = ilqr_nodes[0]->control_dim();

    std::vector<Eigen::MatrixXd> A(horizon, Eigen::MatrixXd(state_dim, state_dim));
    std::vector<Eigen::MatrixXd> B(horizon, Eigen::MatrixXd(state_dim, control_dim));
    std::vector<Eigen::VectorXd> cost_Jx(horizon + 1, Eigen::VectorXd(state_dim));
    std::vector<Eigen::VectorXd> cost_Ju(horizon, Eigen::VectorXd(control_dim));
    std::vector<Eigen::MatrixXd> cost_Hx(horizon + 1, Eigen::MatrixXd(state_dim, state_dim));
    std::vector<Eigen::MatrixXd> cost_Hu(horizon, Eigen::MatrixXd(control_dim, control_dim));

    for (int t = 0; t < horizon; ++t) {
        ILQRNode* node = ilqr_nodes[t];
        auto dynamics_jacobian = node->dynamicsJacobian(node->state(), node->control());
        A[t] = dynamics_jacobian.first;
        B[t] = dynamics_jacobian.second;
        auto cost_jacobian = node->costJacobian();
        cost_Jx[t] = cost_jacobian.first;
        cost_Ju[t] = cost_jacobian.second;
        auto cost_hessian = node->costHessian();
        cost_Hx[t] = cost_hessian.first;
        cost_Hu[t] = cost_hessian.second;
    }

    auto cost_jacobian = ilqr_nodes[horizon]->costJacobian();
    cost_Jx[horizon] = cost_jacobian.first;
    auto cost_hessian = ilqr_nodes[horizon]->costHessian();
    cost_Hx[horizon] = cost_hessian.first;

    Eigen::VectorXd Vx = cost_Jx[horizon];
    Eigen::MatrixXd Vxx = cost_Hx[horizon];

    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(control_dim, state_dim * horizon);
    Eigen::MatrixXd k = Eigen::MatrixXd::Zero(control_dim, horizon);

    for (int t = horizon - 1; t >= 0; --t) {

        Eigen::VectorXd Qu = cost_Ju[t] + B[t].transpose() * Vx;
        Eigen::VectorXd Qx = cost_Jx[t] + A[t].transpose() * Vx;
        Eigen::MatrixXd Qux = B[t].transpose() * Vxx * A[t];
        Eigen::MatrixXd Quu = cost_Hu[t] + B[t].transpose() * Vxx * B[t];
        Eigen::MatrixXd Qxx = cost_Hx[t] + A[t].transpose() * Vxx * A[t];

        Eigen::MatrixXd Quu_inv = (Quu + Eigen::MatrixXd::Identity(control_dim, control_dim) * 1e-9).inverse();

        K.block(0, t * state_dim, control_dim, state_dim) = -Quu_inv * Qux;
        k.col(t) = -Quu_inv * Qu;

        Vx = Qx + K.block(0, t * state_dim, control_dim, state_dim).transpose() * Quu * k.col(t) + K.block(0, t * state_dim, control_dim, state_dim).transpose() * Qu + Qux.transpose() * k.col(t);
        Vxx = Qxx + K.block(0, t * state_dim, control_dim, state_dim).transpose() * Quu * K.block(0, t * state_dim, control_dim, state_dim) + K.block(0, t * state_dim, control_dim, state_dim).transpose() * Qux + Qux.transpose() * K.block(0, t * state_dim, control_dim, state_dim);
    }
    return {k, K};
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> ALILQR::forward(const Eigen::MatrixXd& k, const Eigen::MatrixXd& K) {    
    double alpha = 1.0;

    int state_dim = ilqr_nodes[0]->state_dim();
    int control_dim = ilqr_nodes[0]->control_dim();

    Eigen::MatrixXd x = Eigen::MatrixXd(state_dim, horizon + 1);
    Eigen::MatrixXd u = Eigen::MatrixXd(control_dim, horizon);

    for (int i = 0; i <= horizon; ++i) {
        x.col(i) = ilqr_nodes[i]->state();
    }

    for (int i = 0; i < horizon; ++i) {
        u.col(i) = ilqr_nodes[i]->control();
    }

    Eigen::MatrixXd new_x = Eigen::MatrixXd::Zero(state_dim, horizon + 1);
    Eigen::MatrixXd new_u = Eigen::MatrixXd::Zero(control_dim, horizon);

    new_x.col(0) = x.col(0);
    double old_cost = computeTotalCost();

    while (alpha > 1e-8) {
        auto start_alpha = high_resolution_clock::now();

        for (int t = 0; t < horizon; ++t) {
            new_u.col(t) = u.col(t) + alpha * k.col(t) + K.block(0, t * state_dim, control_dim, state_dim) * (new_x.col(t) - x.col(t));
            new_x.col(t + 1) = ilqr_nodes[t]->dynamics(new_x.col(t), new_u.col(t));
            ilqr_nodes[t + 1]->setState(new_x.col(t + 1));
            ilqr_nodes[t]->setControl(new_u.col(t));
        }
        double new_cost = computeTotalCost();
            
        if (new_cost < old_cost) {
            std::cout << "alpha : " << alpha << std::endl;
            break;
        } else {
            alpha /= 2.0;
        }

        auto end_alpha = high_resolution_clock::now();
        duration<double> alpha_elapsed = end_alpha - start_alpha;
        //std::cout << "forward alpha iteration took " << alpha_elapsed.count() << " seconds." << std::endl;
    }

    if (alpha <= 1e-8) {
        new_x = x;
        new_u = u;
        for (int t = 0; t < horizon; ++t) {
            ilqr_nodes[t + 1]->setState(new_x.col(t + 1));
            ilqr_nodes[t]->setControl(new_u.col(t));
        }
    }

    return {new_x, new_u};
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> ALILQR::optimize(int max_iters, double tol) {
    auto start_total = high_resolution_clock::now();

    auto linearized_initial_guess = linearizedInitialGuess();
    Eigen::MatrixXd x_init = linearized_initial_guess.first;
    Eigen::MatrixXd u_init = linearized_initial_guess.second;
    double old_cost = computeTotalCost();
    Eigen::MatrixXd x = x_init;
    Eigen::MatrixXd u = u_init;

    for (int j = 0; j < 10; ++j) {
        old_cost = computeTotalCost();
        for (int i = 0; i < max_iters; ++i) {
            auto backward_result = backward();

            Eigen::MatrixXd k = backward_result.first;
            Eigen::MatrixXd K = backward_result.second;

            auto forward_result = forward(k, K);

            Eigen::MatrixXd new_x = forward_result.first;
            Eigen::MatrixXd new_u = forward_result.second;
            double new_cost = computeTotalCost();
            if (std::abs(new_cost - old_cost) < tol) {
                break;
            }
            x = new_x;
            u = new_u;
            old_cost = new_cost;
        }
        double violation = computeConstraintViolation();
        if (violation < 1e-2) {
            break;
        } else if (violation >= 1e-2 && violation < 1e-1) {
            updateLambda();
        } else {
            updateMu(4.0);
            //updateLambda();
        }
    }

    auto end_total = high_resolution_clock::now();
    duration<double> total_elapsed = end_total - start_total;
    std::cout << "Total optimization took " << total_elapsed.count() << " seconds." << std::endl;

    return std::make_tuple(x_init, u_init, x, u);
}

void ALILQR::updateLambda() {
    for (ILQRNode* node : ilqr_nodes) {
        node->updateLambda();
    }
}

void ALILQR::updateMu(double gain) {
    for (ILQRNode* node : ilqr_nodes) {
        node->setMu(node->mu() * gain);
    }
}

double ALILQR::computeConstraintViolation() {
    double violation = 0.0;
    for (ILQRNode* node : ilqr_nodes) {
        Eigen::VectorXd one_constrain = node->constraints().cwiseMax(-node->lambda() / node->mu());
        double one_violation = one_constrain.transpose() * one_constrain;
        violation += std::sqrt(one_violation);
    }
    return violation;
}
