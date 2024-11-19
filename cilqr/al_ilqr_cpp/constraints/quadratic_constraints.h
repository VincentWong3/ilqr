#ifndef CONSTRAINTS_QUADRATIC_CONSTRAINTS_H_
#define CONSTRAINTS_QUADRATIC_CONSTRAINTS_H_

#include "constraints.h"

template <int state_dim, int control_dim, int constraint_dim>
class QuadraticConstraints : public Constraints<state_dim, control_dim, constraint_dim> {
public:
    QuadraticConstraints(const std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>& Q,
                      const Eigen::Matrix<double, constraint_dim, state_dim>& A, 
                      const Eigen::Matrix<double, constraint_dim, control_dim>& B, 
                      const Eigen::Matrix<double, constraint_dim, 1>& C, 
                      bool is_equality = false)
        : Constraints<state_dim, control_dim, constraint_dim>(is_equality), Q_(Q), A_(A), B_(B), C_(C) {}
    Eigen::Matrix<double, constraint_dim, 1> constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        // c = xT * Q * x + A * x + B * u + C
        Eigen::Matrix<double, constraint_dim, 1> Ax = A_ * x;
        Eigen::Matrix<double, constraint_dim, 1> Bu = B_ * u;
        Eigen::Matrix<double, constraint_dim, 1> quadratic_term;
        Eigen::Matrix<double, 1, state_dim> xt = x.transpose();
        for (size_t i = 0; i < constraint_dim; ++i) {
            quadratic_term[i] = (xt * Q_[i] * x).value();
        }
        
        return quadratic_term + Ax + Bu + C_;
    }

    Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x, 
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const {
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> C_parallel = C_.replicate(1, PARALLEL_NUM);
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> Ax = A_ * x;
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> Bu = B_ * u;
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> quadratic_term;
        quadratic_term.setZero();

        Eigen::Matrix<double, PARALLEL_NUM, state_dim> xt = x.transpose();
        for (int i = 0; i < constraint_dim; ++i) {
            Eigen::Matrix<double, PARALLEL_NUM, PARALLEL_NUM> temp =  xt * Q_[i] * x;
            for (int j = 0; j < PARALLEL_NUM; j++) {
                quadratic_term(i, j) = temp(j, j);
            }
        }
        return Ax + Bu + C_parallel + quadratic_term;
    }

    std::pair<Eigen::Matrix<double, constraint_dim, state_dim>, Eigen::Matrix<double, constraint_dim, control_dim>> 
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {
        Eigen::Matrix<double, constraint_dim, state_dim> Q_x;
        for (int i = 0; i < constraint_dim; ++i) {
            Q_x.row(i) = 2 * Q_[i] * x;
        }
        return std::make_pair(A_ + Q_x, B_);
    }

    void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) override {
        bool exist = (C_.array() == C_rows).any();
        if (exist) {
            return;
        }
        A_.row(this->current_constraints_index_) = A_rows;
        C_[this->current_constraints_index_] =  C_rows;
    }

    std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>
    constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const override {

        Eigen::Matrix<double, control_dim, control_dim> huu_ele;
        Eigen::Matrix<double, state_dim, control_dim> hxu_ele;

        huu_ele.setZero();
        hxu_ele.setZero();

        std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim> huu;
        std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim> hxu;

        std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> hxx;

        std::fill(huu.begin(), huu.end(), huu_ele);
        std::fill(hxu.begin(), hxu.end(), hxu_ele);

        for (int i = 0; i < constraint_dim; ++i) {
            hxx[i] = 2.0 * Q_[i];
        }

        return std::make_tuple(hxx, huu, hxu);
    }

private:
    std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim> Q_;
    Eigen::Matrix<double, constraint_dim, state_dim> A_;
    Eigen::Matrix<double, constraint_dim, control_dim> B_;
    Eigen::Matrix<double, constraint_dim, 1> C_;
};
#endif // CONSTRAINTS_QUADRATIC_CONSTRAINTS_H_
