#ifndef NEW_LAT_BICYCLE_NODE_H_
#define NEW_LAT_BICYCLE_NODE_H_

#include "new_ilqr_node.h"
#include <Eigen/Dense>
#include <memory>
#include <cmath>

template <class ConstraintsType>
class NewLatBicycleNode : public NewILQRNode<4, 1> {
public:
    using VectorState = typename NewILQRNode<4, 1>::VectorState;
    using VectorControl = typename NewILQRNode<4, 1>::VectorControl;
    using MatrixA = typename NewILQRNode<4, 1>::MatrixA;
    using MatrixB = typename NewILQRNode<4, 1>::MatrixB;
    using MatrixQ = typename NewILQRNode<4, 1>::MatrixQ;
    using MatrixR = typename NewILQRNode<4, 1>::MatrixR;

    NewLatBicycleNode(double L, double dt, double k, double v, const VectorState& goal, const MatrixQ& Q, const MatrixR& R, const ConstraintsType& constraints)
        : NewILQRNode<4, 1>(goal), constraints_(constraints), L_(L), dt_(dt), k_(k), v_(v), Q_(Q), R_(R) {}

    void normalize_state(VectorState& state) const {
        this->normalize_angle(state(2)); // Normalize theta
        this->normalize_angle(state(3)); // Normalize delta
    }

    VectorState dynamics(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        VectorState state_next;

        // Runge-Kutta 2nd Order (RK2) method for discretization
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
        double u1 = control(0);

        this->normalize_angle(theta);
        this->normalize_angle(delta);


        x_dot(0) = v_ * std::cos(theta);
        x_dot(1) = v_ * std::sin(theta);
        x_dot(2) = v_ * std::tan(delta) / (L_ * (1 + k_ * v_ * v_));
        x_dot(3) = u1;

        return x_dot;
    }

    Eigen::MatrixXd parallel_dynamics_continuous(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const {
        int state_rows = state.rows();
        int state_cols = state.cols();
        auto theta_list_matrix_raw = state.row(2);
        auto delta_list_matrix_raw = state.row(3);
        auto v_cos_theta_array = v_ * theta_list_matrix_raw.array().cos();
        auto v_sin_theta_array = v_ * theta_list_matrix_raw.array().sin();

        double denom = L_ * (1.0 + k_ * v_ * v_);
        auto v_tan_delta_array_divide_L = v_ * delta_list_matrix_raw.array().tan() / denom;

        Eigen::MatrixXd state_dot(state_rows, state_cols);

        state_dot.row(0) = v_cos_theta_array.matrix();
        state_dot.row(1) = v_sin_theta_array.matrix();
        state_dot.row(2) = v_tan_delta_array_divide_L.matrix();
        state_dot.row(3) = control.row(0);
        return state_dot;
    }


    Eigen::MatrixXd parallel_dynamics(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const override {
        Eigen::MatrixXd state_dot = parallel_dynamics_continuous(state, control);
        Eigen::MatrixXd mid_state = state + state_dot * dt_ * 0.5;
        Eigen::MatrixXd mid_state_dot = parallel_dynamics_continuous(mid_state, control);
        Eigen::MatrixXd next_state = state + mid_state_dot * dt_;
        return next_state;
    }



    std::pair<MatrixA, MatrixB> dynamics_jacobian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        double theta = state[2], delta = state[3], v = v_;
        double u1 = control[0];

        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / (L * (k * v * v + 1));
        double tan_delta = std::tan(delta);
        double tan_delta_mid = std::tan(delta + 0.5 * dt * u1);
        double denom = L * (k * v * v + 1);
        double sin_theta_mid = std::sin(theta_mid);
        double cos_theta_mid = std::cos(theta_mid);
        double tan_squared = tan_delta * tan_delta;
        double tan_mid_squared = tan_delta_mid * tan_delta_mid;

        // Define Jx
        Eigen::MatrixXd Jx(4, 4);
        Jx << 1, 0, -dt * v * sin_theta_mid, -0.5 * dt * dt * v * v * (tan_squared + 1) * sin_theta_mid / denom,
            0, 1, dt * v * cos_theta_mid,  0.5 * dt * dt * v * v * (tan_squared + 1) * cos_theta_mid / denom,
            0, 0, 1, dt * v * (tan_mid_squared + 1) / denom,
            0, 0, 0, 1;

        // Define Ju
        Eigen::MatrixXd Ju(4, 1);
        Ju << 0,
            0,
            0.5 * dt * dt * v * (tan_mid_squared + 1) / denom,
            dt;

        return {Jx, Ju};
    }

    std::tuple<MatrixA, MatrixA, MatrixA> dynamics_hessian_fxx(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        double theta = state[2], delta = state[3], v = v_;
        double dt = dt_, L = L_, k = k_;
        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / (L * (k * v * v + 1));
        double tan_delta = std::tan(delta);
        double tan_delta_sq = tan_delta * tan_delta;
        double denom = L * (k * v * v + 1);
        double cos_theta_mid = std::cos(theta_mid);
        double sin_theta_mid = std::sin(theta_mid);

        // Initialize Hessian matrices for each element of state_next
        MatrixA hessian_x, hessian_y, hessian_theta, hessian_delta;

        // Fill in Hessian for the x component
        hessian_x << 0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, -dt * v * cos_theta_mid, -0.5 * dt * dt * v * v * (tan_delta_sq + 1) * cos_theta_mid / denom,
                    0, 0, -0.5 * dt * dt * v * v * (tan_delta_sq + 1) * cos_theta_mid / denom, -dt * dt * v * v * ((tan_delta_sq + 1) / denom) * (0.5 * v * (tan_delta_sq + 1) * cos_theta_mid / denom + sin_theta_mid * tan_delta);

        // Fill in Hessian for the y component
        hessian_y << 0, 0, 0, 0,
                    0, 0, 0, 0,
                    0, 0, -dt * v * sin_theta_mid, -0.5 * dt * dt * v * v * (tan_delta_sq + 1) * sin_theta_mid / denom,
                    0, 0, -0.5 * dt * dt * v * v * (tan_delta_sq + 1) * sin_theta_mid / denom, dt * dt * v * v * ((tan_delta_sq + 1) / denom) * (cos_theta_mid * tan_delta - 0.5 * v * (tan_delta_sq + 1) * sin_theta_mid / denom);

        // Fill in Hessian for the theta component
        hessian_theta << 0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 0,
                        0, 0, 0, 2 * dt * v * (tan_delta_sq + 1) * tan_delta / denom;

        return {hessian_x, hessian_y, hessian_theta};
    }

    double cost(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        Eigen::Array<double, 4, 1> Q_array = (Q_.diagonal()).array();
        Eigen::Array<double, 4, 1> error_array = state_error.array();
        Eigen::Array<double, 1, 1> R_array = (R_.diagonal()).array();
        Eigen::Array<double, 1, 1> control_array = control.array();
        Eigen::Matrix<double, 4, 1> new_error = (error_array * Q_array).matrix();
        Eigen::Matrix<double, 1, 1> new_control = (R_array * control_array).matrix();
        double state_cost = (new_error.transpose() * state_error).value();
        double control_cost = (new_control.transpose() * control).value();
        double constraints_cost = constraints_.augmented_lagrangian_cost(state, control);

        return state_cost + control_cost + constraints_cost;
    }

    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost(const Eigen::Ref<const Eigen::Matrix<double, 4, PARALLEL_NUM>>& state, 
                                                         const Eigen::Ref<const Eigen::Matrix<double, 1, PARALLEL_NUM>>& control) override {
        Eigen::Matrix<double, 4, PARALLEL_NUM> error = state - this->goal_.replicate(1, PARALLEL_NUM);
        Eigen::Array<double, 4, PARALLEL_NUM> Q_array = (Q_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 4, PARALLEL_NUM> error_array = error.array();
        Eigen::Array<double, 1, PARALLEL_NUM> R_array = (R_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 1, PARALLEL_NUM> control_array = control.array();
        Eigen::Array<double, 4, PARALLEL_NUM> new_error_array = (error_array * Q_array);
        Eigen::Array<double, 1, PARALLEL_NUM> new_control_array = (R_array * control_array);
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_q = (error_array * new_error_array).matrix().colwise().sum().transpose();
        Eigen::Matrix<double, PARALLEL_NUM, 1> cost_r = (new_control_array * control_array).matrix().colwise().sum().transpose();

        Eigen::Matrix<double, PARALLEL_NUM, 1> ans1 = cost_q + cost_r;
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans2 = constraints_.parallel_augmented_lagrangian_cost(state, control);
        return ans1 + ans2;
    }

    std::pair<Eigen::Matrix<double, 4, 1>, Eigen::Matrix<double, 1, 1>> 
    cost_jacobian(const Eigen::Ref<const VectorState>& state,
                  const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        Eigen::Matrix<double, 4, 1> Jx = 2 * Q_ * state_error;
        Eigen::Matrix<double, 1, 1> Ju = 2 * R_ * control;
        auto constraints_jacobian = constraints_.augmented_lagrangian_jacobian(state, control);
        Jx += constraints_jacobian.first;
        Ju += constraints_jacobian.second;

        return {Jx, Ju};
    }

    std::pair<MatrixQ, MatrixR> cost_hessian(const Eigen::Ref<const VectorState>& state,
                                             const Eigen::Ref<const VectorControl>& control) override {
        MatrixQ Hx = 2 * Q_;
        MatrixR Hu = 2 * R_;
        auto constraints_hessian = constraints_.augmented_lagrangian_hessian(state, control);
        Hx += std::get<0>(constraints_hessian);
        Hu += std::get<1>(constraints_hessian);
        return {Hx, Hu};
    }

    void update_lambda(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) override {
        constraints_.update_lambda(state, control);
    }

    void update_mu(double new_mu) {
        constraints_.update_mu(new_mu);
    }

    void reset_lambda() {
        constraints_.reset_lambda();
    }

    void reset_mu() {
        constraints_.set_mu(1.0);
    }

    double max_constraints_violation(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) const override {
        return constraints_.max_violation(state, control);
    }

    void update_constraints(const Eigen::Ref<const Eigen::Matrix<double, 1, 4>> A_rows, double C_rows) override {
        constraints_.UpdateConstraints(A_rows, C_rows);
    }



public:
ConstraintsType constraints_;


protected:
    double L_;
    double dt_;
    double k_;
    double v_;
    MatrixQ Q_;
    MatrixR R_;
    Eigen::Matrix<double, 4, 1> aug_dx_;
    Eigen::Matrix<double, 1, 1> aug_du_;
    Eigen::Matrix<double, 4, 4> aug_dxx_;
    Eigen::Matrix<double, 1, 1> aug_duu_; 
    Eigen::Matrix<double, 4, 1> aug_dxu_;
    double aug_cost_ = 0.0;

};

#endif // NEW_LAT_BICYCLE_NODE_H_
