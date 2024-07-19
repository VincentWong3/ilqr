#ifndef FAST_FULL_BICYCLE_DYNAMIC_NODE_H
#define FAST_FULL_BICYCLE_DYNAMIC_NODE_H

#include "fast_ilqr_node.h"
#include "fast_sine.h"
#include <array>
#include <iostream>
#include <cmath>

class FastFullBicycleDynamicNode : public FastILQRNode<6, 2> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using VectorState = Eigen::Matrix<double, 6, 1>;
    using VectorControl = Eigen::Matrix<double, 2, 1>;
    using VectorConstraints = Eigen::Matrix<double, 2 * (6 + 2), 1>;
    using MatrixA = Eigen::Matrix<double, 6, 6>;
    using MatrixB = Eigen::Matrix<double, 6, 2>;
    using MatrixQ = Eigen::Matrix<double, 6, 6>;
    using MatrixR = Eigen::Matrix<double, 2, 2>;
    using MatrixCx = Eigen::Matrix<double, 2 * (6 + 2), 6>;
    using MatrixCu = Eigen::Matrix<double, 2 * (6 + 2), 2>;
    using MatrixImu = Eigen::Matrix<double, 2 * (6 + 2), 2 * (6 + 2)>;

    FastFullBicycleDynamicNode(double L, double dt, double k, const std::array<VectorState, 2>& state_bounds, 
                               const std::array<VectorControl, 2>& control_bounds, const VectorState& goal, 
                               const MatrixQ& Q, const MatrixR& R);

    VectorState dynamics(const VectorState& state, const VectorControl& control) override;
    std::pair<MatrixA, MatrixB> dynamicsJacobian(const VectorState& state, const VectorControl& control) override;
    double cost() override;
    std::pair<VectorState, VectorControl> costJacobian() override;
    std::pair<MatrixQ, MatrixR> costHessian() override;
    std::pair<MatrixCx, MatrixCu> constraintJacobian() override;
    VectorConstraints constraints() override;
    void updateImu();
    void updateLambda() override;

    VectorControl control_max() const override;
    VectorControl control_min() const override;

private:
    VectorState dynamicsContinuous(const VectorState& state, const VectorControl& control);

    double L_;
    double dt_;
    double k_;
    VectorState state_max_;
    VectorState state_min_;
    VectorControl control_max_;
    VectorControl control_min_;
    MatrixQ Q_;
    MatrixR R_;
    MatrixImu Imu_;

    // 临时变量
    VectorState dx_;
    MatrixA Jx_;
    MatrixB Ju_;
    VectorState Cost_x_;
    VectorControl Cost_u_;
    MatrixQ Hx_;
    MatrixR Hu_;
    Eigen::Matrix<double, 12, 6> state_jacobian_;
    Eigen::Matrix<double, 4, 2> control_jacobian_;
    MatrixCx Cx_;
    MatrixCu Cu_;
    Eigen::Matrix<double, 12, 1> state_constraints_;
    Eigen::Matrix<double, 4, 1> control_constraints_;
    VectorConstraints all_constraints_;
};

#endif // FAST_FULL_BICYCLE_DYNAMIC_NODE_H
