#include "fast_full_bicycle_dynamic_node.h"

FastFullBicycleDynamicNode::FastFullBicycleDynamicNode(double L, double dt, double k, const std::array<VectorState, 2>& state_bounds, 
                                                       const std::array<VectorControl, 2>& control_bounds, const VectorState& goal, 
                                                       const MatrixQ& Q, const MatrixR& R)
    : FastILQRNode<6, 2>(goal), L_(L), dt_(dt), k_(k), 
      state_max_(state_bounds[1]), state_min_(state_bounds[0]), 
      control_max_(control_bounds[1]), control_min_(control_bounds[0]), Q_(Q), R_(R),
      Imu_(MatrixImu::Zero()) {}

FastFullBicycleDynamicNode::VectorState 
FastFullBicycleDynamicNode::dynamics(const VectorState& state, const VectorControl& control) {
    // Step 1: Calculate k1
    VectorState k1 = dynamicsContinuous(state, control);
    
    // Step 2: Calculate k2
    VectorState mid_state1 = state + 0.5 * dt_ * k1;
    VectorState k2 = dynamicsContinuous(mid_state1, control);
    
    // Step 3: Calculate k3
    VectorState mid_state2 = state + 0.5 * dt_ * k2;
    VectorState k3 = dynamicsContinuous(mid_state2, control);
    
    // Step 4: Calculate k4
    VectorState end_state = state + dt_ * k3;
    VectorState k4 = dynamicsContinuous(end_state, control);
    
    // Combine all increments to get the next state
    //VectorState state_next = state + (dt_ / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    VectorState state_next = state + dt_ * k2;
    
    // Normalize theta and delta
    state_next[2] = this->normalizeAngle(state_next[2]);  // Normalize theta
    state_next[3] = this->normalizeAngle(state_next[3]);  // Normalize delta
    return state_next;
}

FastFullBicycleDynamicNode::VectorState 
FastFullBicycleDynamicNode::dynamicsContinuous(const VectorState& state, const VectorControl& control) {
    double theta = state[2], delta = state[3], v = state[4], a = state[5];
    double u1 = control[0], u2 = control[1];
    theta = this->normalizeAngle(theta);
    delta = this->normalizeAngle(delta);

    double x_dot = v * std::cos(theta);
    double y_dot = v * std::sin(theta);

    double theta_dot = v * std::tan(delta) / (L_ * (1 + k_ * v * v));
    double delta_dot = u1;
    double v_dot = a;
    double a_dot = u2;

    dx_ << x_dot, y_dot, theta_dot, delta_dot, v_dot, a_dot;
    return dx_;
}

std::pair<FastFullBicycleDynamicNode::MatrixA, FastFullBicycleDynamicNode::MatrixB> 
FastFullBicycleDynamicNode::dynamicsJacobian(const VectorState& state, const VectorControl& control) {
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
    Jx_ << 1, 0, -dt * (0.5 * a * dt + v) * sin_theta_mid,
            -0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan_delta_square + 1) * sin_theta_mid / (L * stability_factor), dt * cos_theta_mid - dt * (0.5 * a * dt + v) * (-dt * k * v * v * tan_delta / (L * stability_factor * stability_factor) + 0.5 * dt * tan_delta / (L * stability_factor)) * sin_theta_mid, 0.5 * dt * dt * cos_theta_mid,
            0, 1, dt * (0.5 * a * dt + v) * cos_theta_mid,
            0.5 * dt * dt * v * (0.5 * a * dt + v) * (tan_delta_square + 1) * cos_theta_mid / (L * stability_factor), dt * sin_theta_mid + dt * (0.5 * a * dt + v) * (-dt * k * v * v * tan_delta / (L * stability_factor * stability_factor) + 0.5 * dt * tan_delta / (L * stability_factor)) * cos_theta_mid, 0.5 * dt * dt * sin_theta_mid,
            0, 0, 1, dt * v_tan_delta_u1 / (L * stability_factor), dt * tan_delta_u1 / (L * stability_factor) - dt * k * (0.5 * a * dt + v) * (a * dt + 2 * v) * tan_delta_u1 / (L * stability_factor * stability_factor), 0.5 * dt * dt * tan_delta_u1 / (L * stability_factor) - 0.5 * dt * dt * k * (0.5 * a * dt + v) * (0.5 * a * dt + v) * tan_delta_u1 / (L * stability_factor * stability_factor),
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, dt,
            0, 0, 0, 0, 0, 1;

    Eigen::MatrixXd Ju(6, 2);
    Ju_ << 0, 0,
            0, 0,
            0.5 * dt * dt * v_tan_delta_u1 / (L * stability_factor), 0,
            dt, 0,
            0, 0.5 * dt * dt,
            0, dt;

    return {Jx_, Ju_};

}

double FastFullBicycleDynamicNode::cost() {
    this->normalizeState();
    VectorState state_error = this->state_ - this->goal_;
    double state_cost = state_error.transpose() * Q_ * state_error;
    double control_cost = this->control_.transpose() * R_ * this->control_;
    updateImu();
    VectorConstraints constraints = this->constraints();
    double constraint_cost = (this->lambda_.transpose() * constraints + 0.5 * this->mu_ * constraints.transpose() * Imu_ * constraints)(0,0);
    return state_cost + control_cost + constraint_cost;
}

std::pair<FastFullBicycleDynamicNode::VectorState, FastFullBicycleDynamicNode::VectorControl> 
FastFullBicycleDynamicNode::costJacobian() {
    this->normalizeState();
    VectorState state_error = this->state_ - this->goal_;
    Cost_x_ = 2 * Q_ * state_error;
    Cost_u_ = 2 * R_ * this->control_;
    updateImu();
    auto constraint_jacobian = constraintJacobian();
    MatrixCx constrain_jx = constraint_jacobian.first;
    MatrixCu constrain_ju = constraint_jacobian.second;
    Cost_x_.noalias() += constrain_jx.transpose() * (this->lambda_ + Imu_ * this->constraints());
    Cost_u_.noalias() += constrain_ju.transpose() * (this->lambda_ + Imu_ * this->constraints());
    return {Cost_x_, Cost_u_};
}

std::pair<FastFullBicycleDynamicNode::MatrixQ, FastFullBicycleDynamicNode::MatrixR> 
FastFullBicycleDynamicNode::costHessian() {
    Hx_.setZero();
    Hx_.noalias() = 2 * Q_;
    Hu_.setZero();
    Hu_.noalias() = 2 * R_;
    updateImu();
    auto constraint_jacobian = constraintJacobian();
    MatrixCx constrain_jx = constraint_jacobian.first;
    MatrixCu constrain_ju = constraint_jacobian.second;
    Hx_.noalias() += constrain_jx.transpose() * Imu_ * constrain_jx;
    Hu_.noalias() += constrain_ju.transpose() * Imu_ * constrain_ju;
    return {Hx_, Hu_};
}

std::pair<FastFullBicycleDynamicNode::MatrixCx, FastFullBicycleDynamicNode::MatrixCu> 
FastFullBicycleDynamicNode::constraintJacobian() {
    this->normalizeState();
    state_jacobian_.setZero();
    state_jacobian_.block(0, 0, 6, 6).noalias() = MatrixA::Identity();
    state_jacobian_.block(6, 0, 6, 6).noalias() = -MatrixA::Identity();

    control_jacobian_.setZero();
    control_jacobian_.block(0, 0, 2, 2).noalias() = Eigen::Matrix<double, 2, 2>::Identity();
    control_jacobian_.block(2, 0, 2, 2).noalias() = -Eigen::Matrix<double, 2, 2>::Identity();

    Cx_.setZero();
    Cx_.block(0, 0, 12, 6).noalias() = state_jacobian_;
    Cu_.block(12, 0, 4, 2).noalias() = control_jacobian_;

    return {Cx_, Cu_};
}

FastFullBicycleDynamicNode::VectorConstraints 
FastFullBicycleDynamicNode::constraints() {
    this->normalizeState();  // 确保状态已标准化

    state_constraints_.setZero();
    state_constraints_.head(6).noalias() = this->state_ - state_max_;
    state_constraints_.segment(6, 6).noalias() = state_min_ - this->state_;

    control_constraints_.setZero();
    control_constraints_.head(2).noalias() = this->control_ - control_max_;
    control_constraints_.segment(2, 2).noalias() = control_min_ - this->control_;

    all_constraints_.setZero();
    all_constraints_.head(state_constraints_.size()).noalias() = state_constraints_;
    all_constraints_.segment(state_constraints_.size(), control_constraints_.size()).noalias() = control_constraints_;

    return all_constraints_;
}

void FastFullBicycleDynamicNode::updateImu() {
    VectorConstraints constraints = this->constraints();
    for (int i = 0; i < 2 * (6 + 2); ++i) {
        if (this->lambda_[i] == 0 && constraints[i] <= 0) {
            Imu_(i, i) = 0;
        } else {
            Imu_(i, i) = this->mu_;
        }
    }
}

void FastFullBicycleDynamicNode::updateLambda() {
    VectorConstraints constraints = this->constraints();
    this->lambda_ = (this->lambda_ + this->mu_ * constraints).cwiseMax(0);
}

FastFullBicycleDynamicNode::VectorControl 
FastFullBicycleDynamicNode::control_max() const { return control_max_; }

FastFullBicycleDynamicNode::VectorControl 
FastFullBicycleDynamicNode::control_min() const { return control_min_; }
