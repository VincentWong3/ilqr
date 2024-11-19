#ifndef CONSTRAINTS_DYNAMIC_LINEAR_CONSTRAINTS_H_
#define CONSTRAINTS_DYNAMIC_LINEAR_CONSTRAINTS_H_

#include "dynamic_constraints.h"

template <int state_dim, int control_dim>
class DynamicLinearConstraints : public DynamicConstraints<state_dim, control_dim> {
public:
    DynamicLinearConstraints(const Eigen::MatrixXd& A, 
                      const Eigen::MatrixXd& B, 
                      const Eigen::VectorXd& C, 
                      bool is_equality = false)
        : DynamicConstraints<state_dim, control_dim>(is_equality), A_(A), B_(B), C_(C) {
        int constraints_dim = C.size();
        Eigen::VectorXd zero(constraints_dim);
        zero.setZero();
        this->set_lambda(zero);
    }

    Eigen::VectorXd constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        Eigen::VectorXd Ax = A_ * x;
        Eigen::VectorXd Bu = B_ * u;
        return Ax + Bu + C_;
    }

    Eigen::MatrixXd parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x, 
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const {
        Eigen::MatrixXd C_parallel = C_.replicate(1, PARALLEL_NUM);
        Eigen::MatrixXd Ax = A_ * x;
        Eigen::MatrixXd Bu = B_ * u;
        return Ax + Bu + C_parallel;
    }

    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        return std::make_pair(A_, B_);
    }

    // std::tuple<std::vector<Eigen::Matrix<double, state_dim, state_dim>>,
    //            std::vector<Eigen::Matrix<double, control_dim, control_dim>>,
    //            std::vector<Eigen::Matrix<double, state_dim, control_dim>>>
    // constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
    //                    const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {

    //     Eigen::Matrix<double, state_dim, state_dim> hxx_ele;
    //     Eigen::Matrix<double, control_dim, control_dim> huu_ele;
    //     Eigen::Matrix<double, state_dim, control_dim> hxu_ele;

    //     hxx_ele.setZero();
    //     huu_ele.setZero();
    //     hxu_ele.setZero();

    //     std::vector<Eigen::Matrix<double, state_dim, state_dim>> hxx;
    //     std::vector<Eigen::Matrix<double, control_dim, control_dim>> huu;
    //     std::vector<Eigen::Matrix<double, state_dim, control_dim>> hxu;

    //     std::fill(hxx.begin(), hxx.end(), hxx_ele);
    //     std::fill(huu.begin(), huu.end(), huu_ele);
    //     std::fill(hxu.begin(), hxu.end(), hxu_ele);
        
    //     return std::make_tuple(hxx, huu, hxu);
    // }

private:
    Eigen::MatrixXd A_;
    Eigen::MatrixXd B_;
    Eigen::VectorXd C_;
};
#endif // CONSTRAINTS_DYNAMIC_LINEAR_CONSTRAINTS_H_
