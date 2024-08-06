#ifndef CONSTRAIN_LINEAR_CONSTRAINTS_H_
#define CONSTRAIN_LINEAR_CONSTRAINTS_H_

#include "constraints.h"

template <int state_dim, int control_dim, int constraint_dim>
class LinearConstraints : public Constraints<state_dim, control_dim, constraint_dim> {
public:
    LinearConstraints(const Eigen::Matrix<double, constraint_dim, state_dim>& A, 
                      const Eigen::Matrix<double, constraint_dim, control_dim>& B, 
                      const Eigen::Matrix<double, constraint_dim, 1>& C, 
                      bool is_equality = false)
        : Constraints<state_dim, control_dim, constraint_dim>(is_equality), A_(A), B_(B), C_(C) {}

    Eigen::Matrix<double, constraint_dim, 1> constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        return A_ * x + B_ * u + C_;
    }

    std::pair<Eigen::Matrix<double, constraint_dim, state_dim>, Eigen::Matrix<double, constraint_dim, control_dim>> 
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        return std::make_pair(A_, B_);
    }

    std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>
    constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {

        Eigen::Matrix<double, state_dim, state_dim> hxx_ele;
        Eigen::Matrix<double, control_dim, control_dim> huu_ele;
        Eigen::Matrix<double, state_dim, control_dim> hxu_ele;

        hxx_ele.setZero();
        huu_ele.setZero();
        hxu_ele.setZero();

        std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> hxx;
        std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim> huu;
        std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim> hxu;

        std::fill(hxx.begin(), hxx.end(), hxx_ele);
        std::fill(huu.begin(), huu.end(), huu_ele);
        std::fill(hxu.begin(), hxu.end(), hxu_ele);
        
        return std::make_tuple(hxx, huu, hxu);
    }

private:
    Eigen::Matrix<double, constraint_dim, state_dim> A_;
    Eigen::Matrix<double, constraint_dim, control_dim> B_;
    Eigen::Matrix<double, constraint_dim, 1> C_;
};
#endif // CONSTRAIN_LINEAR_CONSTRAINTS_H_
