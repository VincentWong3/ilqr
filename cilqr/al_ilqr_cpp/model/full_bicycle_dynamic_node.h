#ifndef FULL_BICYCLE_DYNAMIC_NODE_H
#define FULL_BICYCLE_DYNAMIC_NODE_H

#include "ilqr_node.h"
#include <iostream>

class FullBicycleDynamicNode : public ILQRNode {
public:
    FullBicycleDynamicNode(double L, double dt, double k, const std::vector<Eigen::VectorXd>& state_bounds, 
                           const std::vector<Eigen::VectorXd>& control_bounds, const Eigen::VectorXd& goal, 
                           const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
        : ILQRNode(6, 2, 2 * (6 + 2), goal), L_(L), dt_(dt), k_(k), 
          state_max_(state_bounds[1]), state_min_(state_bounds[0]), 
          control_max_(control_bounds[1]), control_min_(control_bounds[0]), Q_(Q), R_(R),
          Imu_(Eigen::MatrixXd::Zero(2 * (6 + 2), 2 * (6 + 2))) {}

    Eigen::VectorXd dynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) override {
        Eigen::VectorXd k1 = dynamicsContinuous(state, control);
        Eigen::VectorXd mid_state = state + 0.5 * dt_ * k1;
        Eigen::VectorXd k2 = dynamicsContinuous(mid_state, control);
        Eigen::VectorXd state_next = state + dt_ * k2;
        state_next[2] = normalizeAngle(state_next[2]);  // Normalize theta
        state_next[3] = normalizeAngle(state_next[3]);  // Normalize delta
        return state_next;
    }

    Eigen::VectorXd dynamicsContinuous(const Eigen::VectorXd& state, const Eigen::VectorXd& control) {
        double theta = state[2], delta = state[3], v = state[4], a = state[5];
        double u1 = control[0], u2 = control[1];
        theta = normalizeAngle(theta);
        delta = normalizeAngle(delta);

        double x_dot = v * std::cos(theta);
        double y_dot = v * std::sin(theta);
        double theta_dot = v * std::tan(delta) / (L_ * (1 + k_ * v * v));
        double delta_dot = u1;
        double v_dot = a;
        double a_dot = u2;

        Eigen::VectorXd dx(6);
        dx << x_dot, y_dot, theta_dot, delta_dot, v_dot, a_dot;
        return dx;
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dynamicsJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) override {
        double theta = state[2], delta = state[3], v = state[4], a = state[5];
        double u1 = control[0];

        theta = normalizeAngle(theta);
        delta = normalizeAngle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / (L * (1 + k * v * v));
        double tan_delta = std::tan(delta);
        double tan_delta_u1 = std::tan(delta + 0.5 * dt * u1);
        double cos_theta_mid = std::cos(theta_mid);
        double sin_theta_mid = std::sin(theta_mid);
        double tan_delta_square = tan_delta * tan_delta;
        double tan_delta_u1_square = tan_delta_u1 * tan_delta_u1;
        double v_tan_delta_u1 = v * (tan_delta_u1_square + 1);
        double stability_factor = 1 + k * v * v;

        Eigen::MatrixXd Jx(6, 6);
        Jx << 1, 0, -dt * (0.5 * a * dt + v) * sin_theta_mid,
              -0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan_delta_square + 1) * sin_theta_mid / (L * stability_factor), dt * cos_theta_mid - dt * (0.5 * a * dt + v) * (-dt * k * v * v * tan_delta / (L * stability_factor * stability_factor) + 0.5 * dt * tan_delta / (L * stability_factor)) * sin_theta_mid, 0.5 * dt * dt * cos_theta_mid,
              0, 1, dt * (0.5 * a * dt + v) * cos_theta_mid,
              0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan_delta_square + 1) * cos_theta_mid / (L * stability_factor), dt * sin_theta_mid + dt * (0.5 * a * dt + v) * (-dt * k * v * v * tan_delta / (L * stability_factor * stability_factor) + 0.5 * dt * tan_delta / (L * stability_factor)) * cos_theta_mid, 0.5 * dt * dt * sin_theta_mid,
              0, 0, 1, dt * v_tan_delta_u1 / (L * stability_factor), dt * tan_delta_u1 / (L * stability_factor) - dt * k * (0.5 * a * dt + v) * (a * dt + 2 * v) * tan_delta_u1 / (L * stability_factor * stability_factor), 0.5 * dt * dt * tan_delta_u1 / (L * stability_factor) - 0.5 * dt * dt * k * (0.5 * a * dt + v) * (0.5 * a * dt + v) * tan_delta_u1 / (L * stability_factor * stability_factor),
              0, 0, 0, 1, 0, 0,
              0, 0, 0, 0, 1, dt,
              0, 0, 0, 0, 0, 1;

        Eigen::MatrixXd Ju(6, 2);
        Ju << 0, 0,
              0, 0,
              0.5 * dt * dt * v_tan_delta_u1 / (L * stability_factor), 0,
              dt, 0,
              0, 0.5 * dt * dt,
              0, dt;

        return {Jx, Ju};
    }

    double cost() override {
        normalizeState();
        Eigen::VectorXd state_error = state_ - goal_;
        double state_cost = state_error.transpose() * Q_ * state_error;
        double control_cost = control_.transpose() * R_ * control_;
        updateImu();
        Eigen::VectorXd constraints = this->constraints();
        double constraint_cost = (lambda_.transpose() * constraints + 0.5 * mu_ * constraints.transpose() * Imu_ * constraints)(0,0);
        return state_cost + control_cost + constraint_cost;
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd> costJacobian() override {
        normalizeState();
        Eigen::VectorXd state_error = state_ - goal_;
        Eigen::VectorXd Jx = 2 * Q_ * state_error;
        Eigen::VectorXd Ju = 2 * R_ * control_;
        updateImu();
        auto constraint_jacobian = constraintJacobian();
        Eigen::MatrixXd constrain_jx = constraint_jacobian.first;
        Eigen::MatrixXd constrain_ju = constraint_jacobian.second;
        Jx += constrain_jx.transpose() * (lambda_ + Imu_ * constraints());
        Ju += constrain_ju.transpose() * (lambda_ + Imu_ * constraints());
        return {Jx, Ju};
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> costHessian() override {
        Eigen::MatrixXd Hx = 2 * Q_;
        Eigen::MatrixXd Hu = 2 * R_;
        updateImu();
        auto constraint_jacobian = constraintJacobian();
        Eigen::MatrixXd constrain_jx = constraint_jacobian.first;
        Eigen::MatrixXd constrain_ju = constraint_jacobian.second;
        Hx += constrain_jx.transpose() * Imu_ * constrain_jx;
        Hu += constrain_ju.transpose() * Imu_ * constrain_ju;
        return {Hx, Hu};
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> constraintJacobian() override {
        normalizeState();
        Eigen::MatrixXd state_jacobian = Eigen::MatrixXd::Zero(2 * state_dim_, state_dim_);
        state_jacobian.block(0, 0, state_dim_, state_dim_) = Eigen::MatrixXd::Identity(state_dim_, state_dim_);
        state_jacobian.block(state_dim_, 0, state_dim_, state_dim_) = -Eigen::MatrixXd::Identity(state_dim_, state_dim_);

        Eigen::MatrixXd control_jacobian = Eigen::MatrixXd::Zero(2 * control_dim_, control_dim_);
        control_jacobian.block(0, 0, control_dim_, control_dim_) = Eigen::MatrixXd::Identity(control_dim_, control_dim_);
        control_jacobian.block(control_dim_, 0, control_dim_, control_dim_) = -Eigen::MatrixXd::Identity(control_dim_, control_dim_);

        Eigen::MatrixXd constraint_jacobian = Eigen::MatrixXd::Zero(constraint_dim_, state_dim_ + control_dim_);
        constraint_jacobian.block(0, 0, 2 * state_dim_, state_dim_) = state_jacobian;
        constraint_jacobian.block(2 * state_dim_, state_dim_, 2 * control_dim_, control_dim_) = control_jacobian;

        Eigen::MatrixXd Jx = constraint_jacobian.block(0, 0, constraint_dim_, state_dim_);
        Eigen::MatrixXd Ju = constraint_jacobian.block(0, state_dim_, constraint_dim_, control_dim_);

        return {Jx, Ju};
    }


    Eigen::VectorXd constraints() override {
        normalizeState();  // 确保状态已标准化

        Eigen::VectorXd state_constraints(state_dim_ * 2);
        state_constraints.head(state_dim_) = state_ - state_max_;
        state_constraints.tail(state_dim_) = state_min_ - state_;


        Eigen::VectorXd control_constraints(control_dim_ * 2);
        control_constraints.head(control_dim_) = control_ - control_max_;
        control_constraints.tail(control_dim_) = control_min_ - control_;

        Eigen::VectorXd all_constraints(constraint_dim_);
        all_constraints.head(state_constraints.size()) = state_constraints;
        all_constraints.tail(control_constraints.size()) = control_constraints;

        return all_constraints;
    }

    void updateImu() {
        Eigen::VectorXd constraints = this->constraints();
        for (int i = 0; i < constraint_dim_; ++i) {
            if (lambda_[i] == 0 && constraints[i] <= 0) {
                Imu_(i, i) = 0;
            } else {
                Imu_(i, i) = mu_;
            }
        }
    }

    void updateLambda() override {
        Eigen::VectorXd constraints = this->constraints();
        lambda_ = (lambda_ + mu_ * constraints).cwiseMax(0);
    }

    Eigen::VectorXd control_max() const override { return control_max_; }
    Eigen::VectorXd control_min() const override { return control_min_; }

private:
    double L_;
    double dt_;
    double k_;
    Eigen::VectorXd state_max_;
    Eigen::VectorXd state_min_;
    Eigen::VectorXd control_max_;
    Eigen::VectorXd control_min_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd R_;
    Eigen::MatrixXd Imu_;
};

#endif // FULL_BICYCLE_DYNAMIC_NODE_H
