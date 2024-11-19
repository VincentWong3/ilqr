#ifndef NEW_ILQR_NODE_H_
#define NEW_ILQR_NODE_H_

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include "constraints/constraints.h"
#include "parallel_compution_function.h"

template <int state_dim, int control_dim>
class NewILQRNode {
public:
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using VectorControl = Eigen::Matrix<double, control_dim, 1>;
    using MatrixA = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixB = Eigen::Matrix<double, state_dim, control_dim>;
    using MatrixQ = Eigen::Matrix<double, state_dim, state_dim>;
    using MatrixR = Eigen::Matrix<double, control_dim, control_dim>;

    explicit NewILQRNode(const VectorState& goal) : goal_(goal) {}

    virtual ~NewILQRNode() = default;

    static constexpr int get_state_dim() {
        return state_dim;
    }

    static constexpr int get_control_dim() {
        return control_dim;
    }

    void normalize_angle(double& angle) const {
        if (std::fabs(angle) < M_PI) {
            return;
        }
        angle = std::fmod(angle + M_PI, 2 * M_PI) - M_PI;
    }

    // Pure virtual functions to be implemented by derived classes
    virtual VectorState dynamics(const Eigen::Ref<const VectorState>& state,
                                 const Eigen::Ref<const VectorControl>& control) const = 0;

    virtual Eigen::MatrixXd parallel_dynamics(const Eigen::Ref<const Eigen::MatrixXd>& state, const Eigen::Ref<const Eigen::MatrixXd>& control) const = 0;

    virtual double cost(const Eigen::Ref<const VectorState>& state, const Eigen::Ref<const VectorControl>& control) = 0;

    virtual Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& state, 
                                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& control) = 0;

    virtual std::pair<MatrixA, MatrixB> dynamics_jacobian(const Eigen::Ref<const VectorState>& state,
                                                          const Eigen::Ref<const VectorControl>& control) const = 0;

    virtual std::tuple<MatrixA, MatrixA, MatrixA> dynamics_hessian_fxx(const Eigen::Ref<const VectorState>& state,
                                                                       const Eigen::Ref<const VectorControl>& control) const = 0;

    virtual std::pair<Eigen::Matrix<double, state_dim, 1>, Eigen::Matrix<double, control_dim, 1>> 
    cost_jacobian(const Eigen::Ref<const VectorState>& state,
                  const Eigen::Ref<const VectorControl>& control) = 0;

    virtual std::pair<MatrixQ, MatrixR> cost_hessian(const Eigen::Ref<const VectorState>& state,
                                                     const Eigen::Ref<const VectorControl>& control) = 0;

    virtual void update_lambda(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) = 0;

    virtual void update_mu(double new_mu) = 0;

    virtual void reset_lambda() = 0;

    virtual void reset_mu() = 0;

    virtual void update_constraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) {
        
    }

    virtual double max_constraints_violation(const Eigen::Ref<const VectorState>& state,
                               const Eigen::Ref<const VectorControl>& control) const = 0;

    

    VectorState goal() { return goal_; }

protected:
    VectorState goal_;
};
#endif // NEW_ILQR_NODE_H_
