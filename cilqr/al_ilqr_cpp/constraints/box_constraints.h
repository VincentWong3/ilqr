#ifndef CONSTRAINTS_BOX_CONSTRAINTS_H_
#define CONSTRAINTS_BOX_CONSTRAINTS_H_

#include "linear_constraints.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>

// BoxConstraints 类
template <int state_dim, int control_dim>
class BoxConstraints : public LinearConstraints<state_dim, control_dim, 2 * (state_dim + control_dim)> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    BoxConstraints(const Eigen::Matrix<double, state_dim, 1>& state_min,
                   const Eigen::Matrix<double, state_dim, 1>& state_max,
                   const Eigen::Matrix<double, control_dim, 1>& control_min,
                   const Eigen::Matrix<double, control_dim, 1>& control_max)
        : LinearConstraints<state_dim, control_dim, 2 * (state_dim + control_dim)>(generateA(state_min, state_max),
                                                                                   generateB(control_min, control_max),
                                                                                   generateC(state_min, state_max, control_min, control_max)) {}
    void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) override {
        bool exist = (C_.array() == C_rows).any();
        if (exist) {
            return;
        }
        this->current_constraints_index_ = this->current_constraints_index_ + 1;
        this->A_.row(this->current_constraints_index_) = A_rows;
        this->C_[this->current_constraints_index_] =  C_rows;
    }

private:
    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, state_dim> generateA(const Eigen::Matrix<double, state_dim, 1>& state_min,
                                                                                       const Eigen::Matrix<double, state_dim, 1>& state_max) {
        Eigen::Matrix<double, 2 * state_dim, state_dim> A_state;
        A_state << Eigen::Matrix<double, state_dim, state_dim>::Identity(),
                   -Eigen::Matrix<double, state_dim, state_dim>::Identity();
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, state_dim> A;
        A << A_state,
             Eigen::Matrix<double, 2 * control_dim, state_dim>::Zero();
        return A;
    }

    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, control_dim> generateB(const Eigen::Matrix<double, control_dim, 1>& control_min,
                                                                                       const Eigen::Matrix<double, control_dim, 1>& control_max) {
        Eigen::Matrix<double, 2 * control_dim, control_dim> B_control;
        B_control << Eigen::Matrix<double, control_dim, control_dim>::Identity(),
                     -Eigen::Matrix<double, control_dim, control_dim>::Identity();
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, control_dim> B;
        B << Eigen::Matrix<double, 2 * state_dim, control_dim>::Zero(),
             B_control;
        return B;
    }

    static Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, 1> generateC(const Eigen::Matrix<double, state_dim, 1>& state_min,
                                                                              const Eigen::Matrix<double, state_dim, 1>& state_max,
                                                                              const Eigen::Matrix<double, control_dim, 1>& control_min,
                                                                              const Eigen::Matrix<double, control_dim, 1>& control_max) {
        Eigen::Matrix<double, 2 * state_dim, 1> C_state;
        C_state << -state_max,
                    state_min;
        Eigen::Matrix<double, 2 * control_dim, 1> C_control;
        C_control << -control_max,
                     control_min;
        Eigen::Matrix<double, 2 * state_dim + 2 * control_dim, 1> C;
        C << C_state,
             C_control;
        return C;
    }

    
};
#endif // CONSTRAINTS_BOX_CONSTRAINTS_H_
