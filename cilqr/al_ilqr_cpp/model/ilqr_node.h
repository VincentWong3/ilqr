#ifndef ILQRNODE_H
#define ILQRNODE_H

#include <Eigen/Dense>
#include <stdexcept>
#include <vector>

class ILQRNode {
public:
    ILQRNode(int state_dim, int control_dim, int constraint_dim, const Eigen::VectorXd& goal)
        : state_dim_(state_dim), control_dim_(control_dim), constraint_dim_(constraint_dim), 
          state_(Eigen::VectorXd::Zero(state_dim)), control_(Eigen::VectorXd::Zero(control_dim)), 
          constraints_(Eigen::VectorXd::Zero(constraint_dim)), goal_(goal.size() == 0 ? Eigen::VectorXd::Zero(state_dim) : goal),
          lambda_(Eigen::VectorXd::Zero(constraint_dim)), mu_(1.0) {}

    virtual ~ILQRNode() = default;

    int state_dim() const { return state_dim_; }
    int control_dim() const { return control_dim_; }
    int constraint_dim() const { return constraint_dim_; }

    Eigen::VectorXd state() const { return state_; }
    void setState(const Eigen::VectorXd& value) {
        if (value.size() != state_dim_) {
            throw std::invalid_argument("Invalid state dimension");
        }
        state_ = value;
    }

    void setState(int index, double value) {
        if (index < 0 || index >= state_dim_) {
            throw std::invalid_argument("Invalid state index");
        }
        state_(index) = value;
    }

    Eigen::VectorXd control() const { return control_; }
    void setControl(const Eigen::VectorXd& value) {
        if (value.size() != control_dim_) {
            throw std::invalid_argument("Invalid control dimension");
        }
        control_ = value;
    }

    Eigen::VectorXd goal() const { return goal_; }
    void setGoal(const Eigen::VectorXd& value) {
        if (value.size() != state_dim_) {
            throw std::invalid_argument("Invalid goal dimension");
        }
        goal_ = value;
    }

    Eigen::VectorXd lambda() const { return lambda_; }
    void setLambda(const Eigen::VectorXd& value) {
        if (value.size() != constraint_dim_) {
            throw std::invalid_argument("Invalid lambda dimension");
        }
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

    virtual Eigen::VectorXd dynamics(const Eigen::VectorXd& state, const Eigen::VectorXd& control) = 0;
    virtual double cost() = 0;
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> dynamicsJacobian(const Eigen::VectorXd& state, const Eigen::VectorXd& control) = 0;
    virtual std::pair<Eigen::VectorXd, Eigen::VectorXd> costJacobian() = 0;
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> costHessian() = 0;
    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> constraintJacobian() = 0;
    virtual Eigen::VectorXd control_max() const = 0;
    virtual Eigen::VectorXd control_min() const = 0;
    virtual Eigen::VectorXd constraints() = 0;

    virtual void updateLambda() = 0;


protected:
    int state_dim_;
    int control_dim_;
    int constraint_dim_;
    Eigen::VectorXd state_;
    Eigen::VectorXd control_;
    Eigen::VectorXd constraints_;
    Eigen::VectorXd goal_;
    Eigen::VectorXd lambda_;
    double mu_;
};

#endif // ILQRNODE_H
