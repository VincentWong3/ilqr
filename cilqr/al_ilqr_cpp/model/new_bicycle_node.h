#ifndef NEW_BICYCLE_NODE_H_
#define NEW_BICYCLE_NODE_H_

#include "new_ilqr_node.h"
#include <Eigen/Dense>
#include <memory>
#include <cmath>

template <class ConstraintsType>
class NewBicycleNode : public NewILQRNode<6, 2> {
public:
    using VectorState = typename NewILQRNode<6, 2>::VectorState;
    using VectorControl = typename NewILQRNode<6, 2>::VectorControl;
    using MatrixA = typename NewILQRNode<6, 2>::MatrixA;
    using MatrixB = typename NewILQRNode<6, 2>::MatrixB;
    using MatrixQ = typename NewILQRNode<6, 2>::MatrixQ;
    using MatrixR = typename NewILQRNode<6, 2>::MatrixR;

    NewBicycleNode(double L, double dt, double k, const VectorState& goal, const MatrixQ& Q, const MatrixR& R, const ConstraintsType& constraints)
        : NewILQRNode<6, 2>(goal), constraints_(constraints), L_(L), dt_(dt), k_(k), Q_(Q), R_(R) {}

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
        double v = state(4);
        double a = state(5);
        double u1 = control(0);
        double u2 = control(1);

        this->normalize_angle(theta);
        this->normalize_angle(delta);

        x_dot(0) = v * std::cos(theta);
        x_dot(1) = v * std::sin(theta);
        x_dot(2) = v * std::tan(delta) / (L_ * (1 + k_ * v * v));
        x_dot(3) = u1;
        x_dot(4) = a;
        x_dot(5) = u2;

        return x_dot;
    }


    std::pair<MatrixA, MatrixB> dynamics_jacobian(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        double theta = state[2], delta = state[3], v = state[4], a = state[5];
        double u1 = control[0];

        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        double theta_mid = theta + 0.5 * dt * v * std::tan(delta) / (L * (k * v * v + 1));
        double v_term = 0.5 * a * dt + v;
        double tan_delta = std::tan(delta);
        double tan_delta_mid = std::tan(delta + 0.5 * dt * u1);
        double k_v_sq = k * v * v;
        double k_v_mid_sq = k * v_term * v_term;
        double denom = L * (k_v_sq + 1);
        double denom_mid = L * (k_v_mid_sq + 1);
        double cos_theta_mid = std::cos(theta_mid);
        double sin_theta_mid = std::sin(theta_mid);

        

        // Define Jx
        Eigen::MatrixXd Jx(6, 6);
        Jx << 1, 0, -dt * (0.5 * a * dt + v) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          -0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan(delta) * tan(delta) + 1) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) / (L * (k * v * v + 1)),
          -dt * (0.5 * a * dt + v) * (-1.0 * dt * k * v * v * tan(delta) / (L * (k * v * v + 1) * (k * v * v + 1)) + 0.5 * dt * tan(delta) / (L * (k * v * v + 1))) * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) + dt * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0, 1, dt * (0.5 * a * dt + v) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan(delta) * tan(delta) + 1) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) / (L * (k * v * v + 1)),
          dt * (0.5 * a * dt + v) * (-1.0 * dt * k * v * v * tan(delta) / (L * (k * v * v + 1) * (k * v * v + 1)) + 0.5 * dt * tan(delta) / (L * (k * v * v + 1))) * cos(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))) + dt * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0.5 * dt * dt * sin(theta + 0.5 * dt * v * tan(delta) / (L * (k * v * v + 1))),
          0, 0, 1, dt * (0.5 * a * dt + v) * (tan(delta + 0.5 * dt * u1) * tan(delta + 0.5 * dt * u1) + 1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          -dt * k * (0.5 * a * dt + v) * (1.0 * a * dt + 2 * v) * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1) * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)) + dt * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          -1.0 * dt * dt * k * (0.5 * a * dt + v) * (0.5 * a * dt + v) * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1) * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)) + 0.5 * dt * dt * tan(delta + 0.5 * dt * u1) / (L * (k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1)),
          0, 0, 0, 1, 0, 0,
          0, 0, 0, 0, 1, dt,
          0, 0, 0, 0, 0, 1;


        // Define Ju
        Eigen::MatrixXd Ju(6, 2);
        Ju << 
            0, 0,
            0, 0,
            0.5 * dt * dt * v_term * (tan_delta_mid * tan_delta_mid + 1) / denom_mid, 0,
            dt, 0,
            0, 0.5 * dt * dt,
            0, dt;

        return {Jx, Ju};
    }

    std::tuple<MatrixA, MatrixA, MatrixA> dynamics_hessian_fxx(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) const override {
        MatrixA ans1, ans2, ans3;
        ans1.setZero();
        ans2.setZero();
        ans3.setZero();
        double theta = state[2], delta = state[3], v = state[4];

        this->normalize_angle(theta);
        this->normalize_angle(delta);

        double dt = dt_;
        double L = L_;
        double k = k_;

        double sin_theta = std::sin(theta);
        double cos_theta = std::cos(theta);
        double tan_delta = std::tan(delta);
        double tan_delta_square_plus_one = tan_delta * tan_delta + 1;
        double k_v_square = k * v * v;
        double k_v_square_plus_one = k_v_square + 1;

        ans1(2, 2) = -dt * v * cos_theta;
        ans1(2, 4) = -dt * sin_theta;
        ans1(4, 2) = ans1(2, 4);

        ans2(2, 2) = -dt * v * sin_theta;
        ans2(2, 4) = dt * cos_theta;
        ans2(4, 2) = ans2(2, 4);

        ans3(3, 3) = 2 * dt * v * tan_delta_square_plus_one * tan_delta / (k_v_square_plus_one * L);
        ans3(3, 4) = dt * (1 - k * v * v) * tan_delta_square_plus_one / (k_v_square_plus_one * L) / (k_v_square_plus_one);
        ans3(4, 3) = ans3(3, 4);
        ans3(4, 4) = dt * 2 * k * v * (k_v_square - 3) * tan_delta / L / k_v_square_plus_one / k_v_square_plus_one / k_v_square_plus_one;

        return {ans1, ans2, ans3};
    }

    double cost(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        Eigen::Array<double, 6, 1> Q_array = (Q_.diagonal()).array();
        Eigen::Array<double, 6, 1> error_array = state_error.array();
        Eigen::Array<double, 2, 1> R_array = (R_.diagonal()).array();
        Eigen::Array<double, 2, 1> control_array = control.array();
        Eigen::Matrix<double, 6, 1> new_error = (error_array * Q_array).matrix();
        Eigen::Matrix<double, 2, 1> new_control = (R_array * control_array).matrix();
        double state_cost = (new_error.transpose() * state_error).value();
        double control_cost = (new_control.transpose() * control).value();
        double constraints_cost = constraints_.augmented_lagrangian_cost(state, control);

        return state_cost + control_cost + constraints_cost;
    }

    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost(const Eigen::Ref<const Eigen::Matrix<double, 6, PARALLEL_NUM>>& state, 
                                                         const Eigen::Ref<const Eigen::Matrix<double, 2, PARALLEL_NUM>>& control) override {
        Eigen::Matrix<double, 6, PARALLEL_NUM> error = state - this->goal_.replicate(1, PARALLEL_NUM);
        Eigen::Array<double, 6, PARALLEL_NUM> Q_array = (Q_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 6, PARALLEL_NUM> error_array = error.array();
        Eigen::Array<double, 2, PARALLEL_NUM> R_array = (R_.diagonal().replicate(1, PARALLEL_NUM)).array();
        Eigen::Array<double, 2, PARALLEL_NUM> control_array = control.array();
        Eigen::Matrix<double, 6, PARALLEL_NUM> new_error = (error_array * Q_array).matrix();
        Eigen::Matrix<double, 2, PARALLEL_NUM> new_control = (R_array * control_array).matrix();
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans1;
        for (int index = 0; index < PARALLEL_NUM; ++index) {
            auto temp1 = new_error.col(index);
            auto temp2 = new_control.col(index);
            ans1[index] = (temp1.transpose() * error.col(index) + temp2.transpose() * control.col(index)).value();
        }
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans2 = constraints_.parallel_augmented_lagrangian_cost(state, control);
        return ans1 + ans2;
    }

    std::pair<Eigen::Matrix<double, 6, 1>, Eigen::Matrix<double, 2, 1>> 
    cost_jacobian(const Eigen::Ref<const VectorState>& state,
                  const Eigen::Ref<const VectorControl>& control) override {
        VectorState state_error = state - this->goal_;
        Eigen::Matrix<double, 6, 1> Jx = 2 * Q_ * state_error;
        Eigen::Matrix<double, 2, 1> Ju = 2 * R_ * control;
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
        auto dim_c = constraints_.get_constraint_dim();
        Eigen::Matrix<double, Eigen::Dynamic, 1> result(dim_c);
        result.setZero();
        constraints_.set_lambda(result);
    }

    void reset_mu() {
        constraints_.set_mu(1.0);
    }

    double max_constraints_violation(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) const override {
        return constraints_.max_violation(state, control);
    }

    void CalcAllCost(const Eigen::Ref<const VectorState>& state,
                             const Eigen::Ref<const VectorControl>& control,
                             double& cost,
                             VectorState& dx,
                             VectorControl& du,
                             MatrixQ& dxx, 
                             MatrixR& duu) override {
        constraints_.CalcAllConstrainInfo(state, control, aug_cost_, aug_dx_, aug_du_, aug_dxx_, aug_duu_, aug_dxu_);
        VectorState state_error = state - this->goal_;
        double state_cost = (state_error.transpose() * Q_ * state_error).value();
        double control_cost = (control.transpose() * R_ * control).value();
        cost = state_cost + control_cost + aug_cost_;
        dx = 2 * Q_ * state_error + aug_dx_;
        du = 2 * R_ * control + aug_du_;
        dxx = 2 * Q_ + aug_dxx_;
        duu = 2 * R_ + aug_duu_;
    }

public:
ConstraintsType constraints_;


protected:
    double L_;
    double dt_;
    double k_;
    MatrixQ Q_;
    MatrixR R_;
    Eigen::Matrix<double, 6, 1> aug_dx_;
    Eigen::Matrix<double, 2, 1> aug_du_;
    Eigen::Matrix<double, 6, 6> aug_dxx_;
    Eigen::Matrix<double, 2, 2> aug_duu_; 
    Eigen::Matrix<double, 6, 2> aug_dxu_;
    double aug_cost_ = 0.0;

};

#endif // NEW_BICYCLE_NODE_H_
