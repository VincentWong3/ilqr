#ifndef NEW_LAT_BICYCLE_NODE_INNER_H_
#define NEW_LAT_BICYCLE_NODE_INNER_H_

#include "new_ilqr_node.h"
#include <Eigen/Dense>
#include <memory>
#include <cmath>

template <class ConstraintsType>
class NewLatBicycleNodeInner : public NewILQRNode<4, 1> {
public:
    using VectorState = typename NewILQRNode<4, 1>::VectorState;
    using VectorControl = typename NewILQRNode<4, 1>::VectorControl;
    using MatrixA = typename NewILQRNode<4, 1>::MatrixA;
    using MatrixB = typename NewILQRNode<4, 1>::MatrixB;
    using MatrixQ = typename NewILQRNode<4, 1>::MatrixQ;
    using MatrixR = typename NewILQRNode<4, 1>::MatrixR;

    NewLatBicycleNodeInner(double L, double dt, double k, double v, double umax, const VectorState& goal, const MatrixQ& Q, const MatrixR& R, const ConstraintsType& constraints)
        : NewILQRNode<4, 1>(goal), constraints_(constraints), L_(L), dt_(dt), k_(k), v_(v), umax_(umax), Q_(Q), R_(R) {}

    void normalize_state(VectorState& state) const {
        this->normalize_angle(state(2)); // Normalize theta
        this->normalize_angle(state(3)); // Normalize delta
    }

    inline double get_u_mapped(double u) const { return umax_ * std::tanh(u); }
    inline double get_du_mapped(double u) const { 
        double th = std::tanh(u);
        return umax_ * (1.0 - th * th); 
    }

    VectorState dynamics(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        VectorState state_next;
        VectorState k1 = dynamics_continuous(state, control);
        VectorState mid_state = state + 0.5 * dt_ * k1;
        VectorState k2 = dynamics_continuous(mid_state, control);

        state_next = state + dt_ * k2;
        normalize_state(state_next);
        return state_next;
    }

    VectorState dynamics_continuous(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const {
        VectorState x_dot;
        double theta = state(2);
        double delta = state(3);
        
        double u_mapped = get_u_mapped(control(0));

        x_dot(0) = v_ * std::cos(theta);
        x_dot(1) = v_ * std::sin(theta);
        x_dot(2) = v_ * std::tan(delta) / (L_ * (1.0 + k_ * v_ * v_));
        x_dot(3) = u_mapped;
        return x_dot;
    }

    Eigen::MatrixXd parallel_dynamics_continuous(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const {
        int state_rows = state.rows();
        int state_cols = state.cols();
        
        auto theta_array = state.row(2).array();
        auto delta_array = state.row(3).array();
        auto u_mapped_array = umax_ * control.row(0).array().tanh();

        double denom = L_ * (1.0 + k_ * v_ * v_);
        Eigen::MatrixXd state_dot(state_rows, state_cols);

        state_dot.row(0) = (v_ * theta_array.cos()).matrix();
        state_dot.row(1) = (v_ * theta_array.sin()).matrix();
        state_dot.row(2) = (v_ * delta_array.tan() / denom).matrix();
        state_dot.row(3) = u_mapped_array.matrix();
        return state_dot;
    }

    Eigen::MatrixXd parallel_dynamics(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const override {
        Eigen::MatrixXd k1 = parallel_dynamics_continuous(state, control);
        Eigen::MatrixXd mid_state = state + k1 * dt_ * 0.5;
        Eigen::MatrixXd k2 = parallel_dynamics_continuous(mid_state, control);
        return state + k2 * dt_;
    }

    std::pair<MatrixA, MatrixB> dynamics_jacobian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        const double theta = state[2];
        const double delta = state[3];
        const double u = control[0];
        const double v = v_;
        const double dt = dt_;
        const double L = L_;
        const double k = k_;

        const double denom = L * (k * v * v + 1.0);
        const double u_mapped = get_u_mapped(u);
        const double du_mapped = get_du_mapped(u);
        
        const double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / denom;
        const double delta_mid = delta + 0.5 * dt * u_mapped;
        
        const double sin_th_mid = std::sin(theta_mid);
        const double cos_th_mid = std::cos(theta_mid);
        const double tan_d = std::tan(delta);
        const double tan_d_sq_plus_1 = tan_d * tan_d + 1.0;
        const double tan_d_mid = std::tan(delta_mid);
        const double tan_d_mid_sq_plus_1 = tan_d_mid * tan_d_mid + 1.0;

        MatrixA Jx = MatrixA::Identity();
        Jx(0, 2) = -dt * v * sin_th_mid;
        Jx(0, 3) = -0.5 * dt * dt * v * v * tan_d_sq_plus_1 * sin_th_mid / denom;
        Jx(1, 2) = dt * v * cos_th_mid;
        Jx(1, 3) = 0.5 * dt * dt * v * v * tan_d_sq_plus_1 * cos_th_mid / denom;
        Jx(2, 3) = dt * v * tan_d_mid_sq_plus_1 / denom;

        MatrixB Ju = MatrixB::Zero();
        Ju(2, 0) = 0.5 * dt * dt * v * tan_d_mid_sq_plus_1 * du_mapped / denom;
        Ju(3, 0) = dt * du_mapped;

        return {Jx, Ju};
    }

    std::tuple<MatrixA, MatrixA, MatrixA> dynamics_hessian_fxx(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        const double theta = state[2];
        const double delta = state[3];
        const double u = control[0];
        const double v = v_;
        const double dt = dt_;
        const double L = L_;
        const double k = k_;
        const double denom = L * (k * v * v + 1.0);
        
        const double u_mapped = get_u_mapped(u);
        const double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / denom;
        const double delta_mid = delta + 0.5 * dt * u_mapped;

        const double sin_th_mid = std::sin(theta_mid);
        const double cos_th_mid = std::cos(theta_mid);
        const double tan_d = std::tan(delta);
        const double tan_d_sq_plus_1 = tan_d * tan_d + 1.0;
        const double tan_d_mid = std::tan(delta_mid);
        const double tan_d_mid_sq_plus_1 = tan_d_mid * tan_d_mid + 1.0;

        MatrixA hessian_x = MatrixA::Zero();
        MatrixA hessian_y = MatrixA::Zero();
        MatrixA hessian_theta = MatrixA::Zero();


        hessian_x(2, 2) = -dt * v * cos_th_mid;
        hessian_x(2, 3) = -0.5 * dt * dt * v * v * tan_d_sq_plus_1 * cos_th_mid / denom;
        hessian_x(3, 2) = hessian_x(2, 3);
        hessian_x(3, 3) = -dt * dt * v * v * (sin_th_mid * tan_d + 0.25 * dt * v * tan_d_sq_plus_1 * cos_th_mid / denom) * tan_d_sq_plus_1 / denom;

        hessian_y(2, 2) = -dt * v * sin_th_mid;
        hessian_y(2, 3) = -0.5 * dt * dt * v * v * tan_d_sq_plus_1 * sin_th_mid / denom;
        hessian_y(3, 2) = hessian_y(2, 3);
        hessian_y(3, 3) = dt * dt * v * v * (cos_th_mid * tan_d - 0.25 * dt * v * tan_d_sq_plus_1 * sin_th_mid / denom) * tan_d_sq_plus_1 / denom;

        hessian_theta(3, 3) = 2.0 * dt * v * tan_d_mid_sq_plus_1 * tan_d_mid / denom;

        return {hessian_x, hessian_y, hessian_theta};
    }

    double cost(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        double state_cost = (state_error.transpose() * Q_ * state_error).value();
        double control_cost = (control.transpose() * R_ * control).value();
        double constraints_cost = constraints_.augmented_lagrangian_cost(state, control);
        return state_cost + control_cost + constraints_cost;
    }

    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost(const Eigen::Ref<const Eigen::Matrix<double, 4, PARALLEL_NUM>>& state, 
                                                         const Eigen::Ref<const Eigen::Matrix<double, 1, PARALLEL_NUM>>& control) override {
        Eigen::Matrix<double, 4, PARALLEL_NUM> error = state - this->goal_.replicate(1, PARALLEL_NUM);
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_q = (error.array() * (Q_ * error).array()).colwise().sum().transpose();
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_r = (control.array() * (R_ * control).array()).colwise().sum().transpose();
        return cost_q + cost_r + constraints_.parallel_augmented_lagrangian_cost(state, control);
    }

    std::pair<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 1, 1>> 
    cost_jacobian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        VectorState Jx = 2.0 * Q_ * (state - this->goal_);
        Eigen::Matrix<double, 1, 1> Ju = 2.0 * R_ * control;
        auto c_jac = constraints_.augmented_lagrangian_jacobian(state, control);
        return {Jx + c_jac.first, Ju + c_jac.second};
    }

    std::pair<MatrixQ, MatrixR> cost_hessian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        auto c_hess = constraints_.augmented_lagrangian_hessian(state, control);
        return {2.0 * Q_ + std::get<0>(c_hess), 2.0 * R_ + std::get<1>(c_hess)};
    }

    void update_lambda(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        constraints_.update_lambda(state, control);
    }
    void update_mu(double new_mu) { constraints_.update_mu(new_mu); }
    void reset_lambda() { constraints_.reset_lambda(); }
    void reset_mu() { constraints_.set_mu(1.0); }
    double max_constraints_violation(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        return constraints_.max_violation(state, control);
    }
    void update_constraints(const Eigen::Ref<const Eigen::Matrix<double, 1, 4>> A_rows, double C_rows) override {
        constraints_.UpdateConstraints(A_rows, C_rows);
    }

public:
    ConstraintsType constraints_;

protected:
    double L_, dt_, k_, v_, umax_;
    MatrixQ Q_;
    MatrixR R_;
};

#endif // NEW_LAT_BICYCLE_NODE_INNER_H_