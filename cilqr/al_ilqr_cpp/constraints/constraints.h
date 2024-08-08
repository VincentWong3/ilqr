#ifndef CONSTRAINTS_CONSTRAINTS_H_
#define CONSTRAINTS_CONSTRAINTS_H_


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <tuple>
#include <utility>
#include <array>
#include <algorithm>

#define PARALLEL_NUM 5

template <int state_dim, int control_dim, int constraint_dim>
class Constraints {
public:
    Constraints(bool is_equality = false)
        : lambda_(Eigen::Matrix<double, constraint_dim, 1>::Zero()),
          mu_(1.0),
          is_equality_(is_equality) {}

    virtual ~Constraints() = default;

    // Getters and Setters
    Eigen::Matrix<double, constraint_dim, 1> lambda() const {
        return lambda_;
    }

    void set_lambda(const Eigen::Matrix<double, constraint_dim, 1>& lambda) {
        lambda_ = lambda;
    }

    double mu() const {
        return mu_;
    }

    void set_mu(double mu) {
        mu_ = mu;
    }

    static constexpr int get_state_dim() {
        return state_dim;
    }

    static constexpr int get_control_dim() {
        return control_dim;
    }

    static constexpr int get_constraint_dim() {
        return constraint_dim;
    }

    virtual Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x, 
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const = 0;

    virtual Eigen::Matrix<double, constraint_dim, 1> constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                                               const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    virtual std::pair<Eigen::Matrix<double, constraint_dim, state_dim>, Eigen::Matrix<double, constraint_dim, control_dim>> 
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    virtual std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>
    constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;


    Eigen::Matrix<double, constraint_dim, 1> projection(const Eigen::Matrix<double, constraint_dim, 1>& x) const {
        return x.cwiseMin(0);
    }

    Eigen::Matrix<double, constraint_dim, constraint_dim> projection_jacobian(const Eigen::Matrix<double, constraint_dim, 1>& x) const {
        Eigen::Matrix<double, constraint_dim, constraint_dim> jac;
        Eigen::Array<double, constraint_dim, 1> x_temp = x.array();
        
        Eigen::Array<double, constraint_dim, 1> ans = (x_temp < 0).template cast<double>();

        return ans.matrix().asDiagonal();
    }

    void projection_jacobian2(const Eigen::Matrix<double, constraint_dim, 1>& x) {
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) >= 0) {
                proj_cx_T_.col(i) = Eigen::Matrix<double, state_dim, 1>::Zero();
                proj_cu_T_.col(i) = Eigen::Matrix<double, control_dim, 1>::Zero();
            }
        }
    }

    Eigen::Matrix<double, constraint_dim, constraint_dim> projection_hessian(const Eigen::Matrix<double, constraint_dim, 1>& x, 
                                                                           const Eigen::Matrix<double, constraint_dim, constraint_dim>& b) const {
        return Eigen::Matrix<double, constraint_dim, constraint_dim>::Zero();
    }

    double augmented_lagrangian_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                     const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);
        c_ = c;
        if (is_equality_) {
            return 0.5 / mu_ * ((lambda_ - mu_ * c).transpose() * (lambda_ - mu_ * c) - lambda_.transpose() * lambda_).value();
        } else {
            lambda_proj_ = projection(lambda_ - mu_ * c);
            return 0.5 / mu_ * (lambda_proj_.transpose() * lambda_proj_ - lambda_.transpose() * lambda_).value();
        }
    }


    Eigen::Matrix<double, PARALLEL_NUM, 1> parallel_augmented_lagrangian_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x, 
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) {
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> c;
        c = parallel_constraints(x, u);
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> parallel_lambda = lambda_.replicate(1, PARALLEL_NUM);
        Eigen::Matrix<double, constraint_dim, PARALLEL_NUM> proj = (parallel_lambda - mu_ * c).cwiseMin(0);
        Eigen::Matrix<double, PARALLEL_NUM, 1> ans;
        for (int index = 0; index < PARALLEL_NUM; ++index) {
            auto temp1 = proj.col(index);
            ans[index] = (temp1.transpose() * temp1 - lambda_.transpose() * lambda_).value();
        }
        ans = 0.5 / mu_ * ans;
       return ans;
    }

    

    std::pair<Eigen::Matrix<double, state_dim, 1>, Eigen::Matrix<double, control_dim, 1>> 
    augmented_lagrangian_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                  const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        auto jacobian_matrix = constraints_jacobian(x, u);
        auto cx = jacobian_matrix.first;
        auto cu = jacobian_matrix.second;
        cx_ = cx;
        cu_ = cu;
        proj_cx_T_ = cx_.transpose();
        proj_cu_T_ = cu_.transpose();
        Eigen::Matrix<double, state_dim, 1> dx;
        Eigen::Matrix<double, control_dim, 1> du;


        if (is_equality_) {
            Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c_;
            dx = -cx.transpose() * factor;
            du = -cu.transpose() * factor;
        } else {
            // proj_jac_ = projection_jacobian(lambda_proj_);
            // proj_cx_T_ = (proj_jac_ * cx).transpose();
            // proj_cu_T_ = (proj_jac_ * cu).transpose();

            projection_jacobian2(lambda_proj_);
            dx = -proj_cx_T_ * lambda_proj_;
            du = -proj_cu_T_ * lambda_proj_;
            dxdx_ = proj_cx_T_ * cx_;
            dudu_ = proj_cu_T_ * cu_;
        }

        return {dx, du};
    }

    std::tuple<Eigen::Matrix<double, state_dim, state_dim>, 
               Eigen::Matrix<double, control_dim, control_dim>, 
               Eigen::Matrix<double, state_dim, control_dim>> 
    augmented_lagrangian_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u, bool full_newton = false) {
        // Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);

        // auto hessian_tensor = constraints_hessian(x, u);

        // auto jacobian_matrix = constraints_jacobian(x, u);
        // auto cx = cx_;
        // auto cu = cu_;
        //Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
        //Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
        Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
        //Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c_;

        
        if (is_equality_) {
            // auto ans = tensor_contract(factor, hessian_tensor);
            // dxdx = mu_ * ((cx.transpose() * cx) - std::get<0>(ans));
            // dxdu = mu_ * ((cx.transpose() * cu) - std::get<2>(ans));
            // dudu = mu_ * ((cu.transpose() * cu) - std::get<1>(ans));
        } else {
            // auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            // dxdx = mu_ * ((jac_proj_cx.transpose() * jac_proj_cx) - std::get<0>(ans));
            // dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
            // dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));

            // auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            //dxdx = (proj_cx_T_ * cx_);
            // dxdu = mu_ * ((cx_.transpose() * proj_jac_ * cu_));
            //dudu = (proj_cu_T_ * cu_);
        }

        return {mu_ * dxdx_, mu_ * dudu_, dxdu};
    }

    void update_lambda(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                       const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        if (is_equality_) {
            lambda_ -= mu_ * constraints(x, u);
        } else {
            lambda_ = projection(lambda_ - mu_ * constraints(x, u));
        }
    }

    void update_mu(double new_mu) {
        mu_ = new_mu;
    }

    double max_violation(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                         const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const {
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);
        Eigen::Matrix<double, constraint_dim, 1> c_proj = projection(c);
        Eigen::Matrix<double, constraint_dim, 1> dc = c - c_proj;
        return dc.template lpNorm<Eigen::Infinity>();
    }

    void CalcAllConstrainInfo(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u,
                 double& augmented_lagrangian_cost,
                 Eigen::Matrix<double, state_dim, 1>& augmented_lagrangian_jacobian_x,
                 Eigen::Matrix<double, control_dim, 1>& augmented_lagrangian_jacobian_u,
                 Eigen::Matrix<double, state_dim, state_dim> & augmented_lagrangian_hessian_xx, 
                 Eigen::Matrix<double, control_dim, control_dim>& augmented_lagrangian_hessian_uu, 
                 Eigen::Matrix<double, state_dim, control_dim>& augmented_lagrangian_hessian_xu) {
        Eigen::Matrix<double, constraint_dim, 1> c = constraints(x, u);
        auto jacobian_matrix = constraints_jacobian(x, u);
        auto cx = jacobian_matrix.first;
        auto cu = jacobian_matrix.second;
        Eigen::Matrix<double, state_dim, 1> dx;
        Eigen::Matrix<double, control_dim, 1> du;
        Eigen::Matrix<double, constraint_dim, 1> factor = lambda_ - mu_ * c;
        Eigen::Matrix<double, constraint_dim, 1> lambda_proj = projection(factor);
        Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
        Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
        Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
        Eigen::Matrix<double, constraint_dim, constraint_dim> proj_jac = projection_jacobian(factor);
        auto hessian_tensor = constraints_hessian(x, u);
        auto ans = tensor_contract(factor, hessian_tensor);

        if (is_equality_) {
            augmented_lagrangian_cost = 0.5 / mu_ * ((lambda_ - mu_ * c).transpose() * (lambda_ - mu_ * c) - lambda_.transpose() * lambda_).value();
            dx = -cx.transpose() * factor;
            du = -cu.transpose() * factor;
            augmented_lagrangian_jacobian_x = dx;
            augmented_lagrangian_jacobian_u = du;
            dxdx = mu_ * ((cx.transpose() * cx) - std::get<0>(ans));
            dxdu = mu_ * ((cx.transpose() * cu) - std::get<2>(ans));
            dudu = mu_ * ((cu.transpose() * cu) - std::get<1>(ans));
            augmented_lagrangian_hessian_xx = dxdx;
            augmented_lagrangian_hessian_uu = dudu;
            augmented_lagrangian_hessian_xu = dxdu;

        } else {
            augmented_lagrangian_cost = 0.5 / mu_ * (lambda_proj.transpose() * lambda_proj - lambda_.transpose() * lambda_).value();
            dx = -(proj_jac * cx).transpose() * lambda_proj;
            du = -(proj_jac * cu).transpose() * lambda_proj;
            Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac * cx;
            Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac * cu;
            dxdx = mu_ * ((jac_proj_cx.transpose() * jac_proj_cx) - std::get<0>(ans));
            dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
            dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));
        }
    }

private:
    Eigen::Matrix<double, constraint_dim, 1> lambda_;
    Eigen::Matrix<double, constraint_dim, 1> lambda_proj_;
    Eigen::Matrix<double, constraint_dim, 1> c_;
    Eigen::Matrix<double, constraint_dim, state_dim> cx_;
    Eigen::Matrix<double, constraint_dim, control_dim> cu_;
    Eigen::Matrix<double, state_dim, constraint_dim> proj_cx_T_;
    Eigen::Matrix<double, control_dim, constraint_dim> proj_cu_T_;
    Eigen::Matrix<double, state_dim, state_dim> dxdx_;
    Eigen::Matrix<double, control_dim, control_dim> dudu_;


    double mu_;
    bool is_equality_;
    Eigen::Matrix<double, constraint_dim, constraint_dim> proj_jac_;




    // Helper function for tensor contraction
    std::tuple<Eigen::Matrix<double, state_dim, state_dim>, Eigen::Matrix<double, control_dim, control_dim>, Eigen::Matrix<double, state_dim, control_dim>>
    tensor_contract(const Eigen::Matrix<double, constraint_dim, 1>& factor, 
                                    const std::tuple<std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, control_dim, control_dim>, constraint_dim>,
                       std::array<Eigen::Matrix<double, state_dim, control_dim>, constraint_dim>>& tensor) const {

        Eigen::Matrix<double, state_dim, state_dim> factor_dot_hxx;
        Eigen::Matrix<double, control_dim, control_dim> factor_dot_huu;
        Eigen::Matrix<double, state_dim, control_dim> factor_dot_hxu;
        auto hxx = std::get<0>(tensor);
        auto huu = std::get<1>(tensor);
        auto hxu = std::get<2>(tensor);

        factor_dot_hxx.setZero();
        factor_dot_huu.setZero();
        factor_dot_hxu.setZero();


        for(size_t index = 0; index < constraint_dim; ++index) {
            factor_dot_hxx += hxx[index] * factor(index, 0);
            factor_dot_huu += huu[index] * factor(index, 0);
            factor_dot_hxu += hxu[index] * factor(index, 0);
        }
        return {factor_dot_hxx, factor_dot_huu, factor_dot_hxu};
    }
};
#endif // CONSTRAIN_CONSTRAIN_H_
