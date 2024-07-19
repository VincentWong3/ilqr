#ifndef FAST_ILQR_NODE_H_
#define FAST_ILQR_NODE_H_

#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>

template<int state_dim, int control_dim>
class FastILQRNode {
public:
    static constexpr int constraint_dim = 2 * (state_dim + control_dim);
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;
    using VectorConstraints = Eigen::Matrix<double, constraint_dim, 1>;
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;
    using MatrixQ = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixR = Eigen::Matrix<double, control_dim, control_dim>;
    using MatrixCx = Eigen::Matrix<double, constraint_dim, state_dim>;
    using MatrixCu = Eigen::Matrix<double, constraint_dim, control_dim>;

    FastILQRNode(const VectorState& goal)
        : state_(VectorState::Zero()), control_(VectorControl::Zero()), constraints_(VectorConstraints::Zero()),
          goal_(goal.size() == 0 ? VectorState::Zero() : goal), lambda_(VectorConstraints::Zero()), mu_(1.0) {}

    virtual ~FastILQRNode() = default;

    VectorState state() const { return state_; }
    void setState(const VectorState& value) {
        state_ = value;
    }

    void setState(int index, double value) {
        if (index < 0 || index >= state_dim) {
            throw std::invalid_argument("Invalid state index");
        }
        state_(index) = value;
    }

    VectorControl control() const { return control_; }
    void setControl(const VectorControl& value) {
        control_ = value;
    }

    VectorState goal() const { return goal_; }
    void setGoal(const VectorState& value) {
        goal_ = value;
    }

    VectorConstraints lambda() const { return lambda_; }
    void setLambda(const VectorConstraints& value) {
        lambda_ = value;
    }

    double mu() const { return mu_; }
    void setMu(double value) { mu_ = value; }

    double normalizeAngle(double angle) const {
        return std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
    }

    void normalizeState() {
        state_[2] = normalizeAngle(state_[2]);  // Normalize theta
        state_[3] = normalizeAngle(state_[3]);  // Normalize delta
    }

    virtual VectorState dynamics(const VectorState& state, const VectorControl& control) = 0;
    virtual double cost() = 0;
    virtual std::pair<MatrixA, MatrixB> dynamicsJacobian(const VectorState& state, const VectorControl& control) = 0;
    virtual std::pair<VectorState, VectorControl> costJacobian() = 0;
    virtual std::pair<MatrixQ, MatrixR> costHessian() = 0;
    virtual std::pair<MatrixCx, MatrixCu> constraintJacobian() = 0;
    virtual VectorControl control_max() const = 0;
    virtual VectorControl control_min() const = 0;
    virtual VectorConstraints constraints() = 0;

    virtual void updateLambda() = 0;

protected:
    VectorState state_;
    VectorControl control_;
    VectorConstraints constraints_;
    VectorState goal_;
    VectorConstraints lambda_;
    double mu_;
};

#endif // FAST_ILQR_NODE_H_
