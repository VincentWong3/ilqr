#ifndef CONSTRAINTS_DYNAMIC_CONSTRAINTS_H_
#define CONSTRAINTS_DYNAMIC_CONSTRAINTS_H_


#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <tuple>
#include <utility>
#include <array>
#include <algorithm>

#define PARALLEL_NUM 5

template <int state_dim, int control_dim>
class DynamicConstraints {
public:
    DynamicConstraints(bool is_equality = false)
        : mu_(1.0),
          is_equality_(is_equality) {}

    virtual ~DynamicConstraints() = default;

    // Getters and Setters
    Eigen::VectorXd lambda() const {
        return lambda_;
    }

    void set_lambda(const Eigen::VectorXd& lambda) {
        lambda_ = lambda;
    }

    void reset_lambda() {
        lambda_.setZero();
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

    virtual Eigen::MatrixXd parallel_constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, PARALLEL_NUM>>& x, 
                                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, PARALLEL_NUM>>& u) const = 0;

    virtual Eigen::VectorXd constraints(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                                               const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd> 
    constraints_jacobian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                        const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;

    // virtual std::tuple<std::vector<Eigen::Matrix<double, state_dim, state_dim>>,
    //                    std::vector<Eigen::Matrix<double, control_dim, control_dim>>,
    //                    std::vector<Eigen::Matrix<double, state_dim, control_dim>>>
    // constraints_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
    //                    const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) const = 0;


    Eigen::VectorXd projection(const Eigen::VectorXd& x) const {
        return x.cwiseMin(0);
    }

    Eigen::MatrixXd projection_jacobian(const Eigen::VectorXd& x) const {

        Eigen::ArrayXd x_temp = x.array();
        
        Eigen::ArrayXd ans = (x_temp < 0).template cast<double>();

        return ans.matrix().asDiagonal();
    }

    void projection_jacobian2(const Eigen::MatrixXd& x) {
        for (int i = 0; i < x.size(); ++i) {
            if (x(i) >= 0) {
                proj_cx_T_.col(i) = Eigen::Matrix<double, state_dim, 1>::Zero();
                proj_cu_T_.col(i) = Eigen::Matrix<double, control_dim, 1>::Zero();
            }
        }
    }

    double augmented_lagrangian_cost(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                     const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u) {
        Eigen::MatrixXd c = constraints(x, u);
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
        Eigen::MatrixXd c = parallel_constraints(x, u);
        Eigen::MatrixXd parallel_lambda = lambda_.replicate(1, PARALLEL_NUM);
        Eigen::MatrixXd proj = (parallel_lambda - mu_ * c).cwiseMin(0);

        Eigen::ArrayXXd parallel_lambda_array = parallel_lambda.array();

        Eigen::ArrayXXd proj_array = proj.array();


        Eigen::MatrixXd ans = (proj_array * proj_array - parallel_lambda_array * parallel_lambda_array).matrix().colwise().sum().transpose();

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
            Eigen::VectorXd factor = lambda_ - mu_ * c_;
            dx = -cx.transpose() * factor;
            du = -cu.transpose() * factor;
        } else {
            // proj_jac_ = projection_jacobian(lambda_proj_);
            // proj_cx_T_ = (proj_jac_ * cx).transpose();
            // proj_cu_T_ = (proj_jac_ * cu).transpose();

            projection_jacobian2(lambda_proj_);
            dx = -proj_cx_T_ * lambda_proj_;
            du = -proj_cu_T_ * lambda_proj_;
            dxdx_ = mu_ * proj_cx_T_ * cx_;
            dudu_ = mu_ * proj_cu_T_ * cu_;
            dxdu_.setZero();
        }

        return {dx, du};
    }

    std::tuple<Eigen::Matrix<double, state_dim, state_dim>, 
               Eigen::Matrix<double, control_dim, control_dim>, 
               Eigen::Matrix<double, state_dim, control_dim>> 
    augmented_lagrangian_hessian(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
                                 const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u, bool full_newton = false) {
        // Eigen::VectorXd c = constraints(x, u);

        // auto hessian_tensor = constraints_hessian(x, u);

        // auto jacobian_matrix = constraints_jacobian(x, u);
        // auto cx = cx_;
        // auto cu = cu_;
        // Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
        // Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
        // Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
        // Eigen::VectorXd factor = lambda_ - mu_ * c_;

        
        if (is_equality_) {
            // auto ans = tensor_contract(factor, hessian_tensor);
            // dxdx_ = mu_ * ((cx_.transpose() * cx_)) - std::get<0>(ans);
            // dxdu_ = mu_ * ((cx_.transpose() * cu_)) - std::get<2>(ans);
            // dudu_ = mu_ * ((cu_.transpose() * cu_)) - std::get<1>(ans);
            dxdx_ = mu_ * ((cx_.transpose() * cx_));
            dxdu_ = mu_ * ((cx_.transpose() * cu_));
            dudu_ = mu_ * ((cu_.transpose() * cu_));
        } else {
            // auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            // dxdx_ =  dxdx_ - 1.0 * std::get<0>(ans);
            // dxdu_ =  dxdu_ - 1.0 * std::get<2>(ans);
            // dudu_ =  dudu_ - 1.0 * std::get<1>(ans);

            // dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
            // dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));

            // auto ans = tensor_contract(lambda_proj_, hessian_tensor);
            // Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac_ * cx;
            // Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac_ * cu;
            //dxdx = (proj_cx_T_ * cx_);
            // dxdu = mu_ * ((cx_.transpose() * proj_jac_ * cu_));
            //dudu = (proj_cu_T_ * cu_);
        }

        return {dxdx_, dudu_, dxdu_};
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
        Eigen::VectorXd c = constraints(x, u);
        Eigen::VectorXd c_proj = projection(c);
        Eigen::VectorXd dc = c - c_proj;
        return dc.template lpNorm<Eigen::Infinity>();
    }

    virtual void UpdateConstraints(const Eigen::Ref<const Eigen::Matrix<double, 1, state_dim>> A_rows, double C_rows) {

    }

    // void CalcAllConstrainInfo(const Eigen::Ref<const Eigen::Matrix<double, state_dim, 1>>& x, 
    //              const Eigen::Ref<const Eigen::Matrix<double, control_dim, 1>>& u,
    //              double& augmented_lagrangian_cost,
    //              Eigen::Matrix<double, state_dim, 1>& augmented_lagrangian_jacobian_x,
    //              Eigen::Matrix<double, control_dim, 1>& augmented_lagrangian_jacobian_u,
    //              Eigen::Matrix<double, state_dim, state_dim> & augmented_lagrangian_hessian_xx, 
    //              Eigen::Matrix<double, control_dim, control_dim>& augmented_lagrangian_hessian_uu, 
    //              Eigen::Matrix<double, state_dim, control_dim>& augmented_lagrangian_hessian_xu) {
    //     Eigen::VectorXd c = constraints(x, u);
    //     auto jacobian_matrix = constraints_jacobian(x, u);
    //     auto cx = jacobian_matrix.first;
    //     auto cu = jacobian_matrix.second;
    //     Eigen::Matrix<double, state_dim, 1> dx;
    //     Eigen::Matrix<double, control_dim, 1> du;
    //     Eigen::VectorXd factor = lambda_ - mu_ * c;
    //     Eigen::VectorXd lambda_proj = projection(factor);
    //     Eigen::Matrix<double, state_dim, state_dim> dxdx = Eigen::Matrix<double, state_dim, state_dim>::Zero();
    //     Eigen::Matrix<double, control_dim, control_dim> dudu = Eigen::Matrix<double, control_dim, control_dim>::Zero();
    //     Eigen::Matrix<double, state_dim, control_dim> dxdu = Eigen::Matrix<double, state_dim, control_dim>::Zero();
    //     Eigen::Matrix<double, constraint_dim, constraint_dim> proj_jac = projection_jacobian(factor);
    //     auto hessian_tensor = constraints_hessian(x, u);
    //     auto ans = tensor_contract(factor, hessian_tensor);

    //     if (is_equality_) {
    //         augmented_lagrangian_cost = 0.5 / mu_ * ((lambda_ - mu_ * c).transpose() * (lambda_ - mu_ * c) - lambda_.transpose() * lambda_).value();
    //         dx = -cx.transpose() * factor;
    //         du = -cu.transpose() * factor;
    //         augmented_lagrangian_jacobian_x = dx;
    //         augmented_lagrangian_jacobian_u = du;
    //         dxdx = mu_ * ((cx.transpose() * cx) - std::get<0>(ans));
    //         dxdu = mu_ * ((cx.transpose() * cu) - std::get<2>(ans));
    //         dudu = mu_ * ((cu.transpose() * cu) - std::get<1>(ans));
    //         augmented_lagrangian_hessian_xx = dxdx;
    //         augmented_lagrangian_hessian_uu = dudu;
    //         augmented_lagrangian_hessian_xu = dxdu;

    //     } else {
    //         augmented_lagrangian_cost = 0.5 / mu_ * (lambda_proj.transpose() * lambda_proj - lambda_.transpose() * lambda_).value();
    //         dx = -(proj_jac * cx).transpose() * lambda_proj;
    //         du = -(proj_jac * cu).transpose() * lambda_proj;
    //         Eigen::Matrix<double, constraint_dim, state_dim> jac_proj_cx = proj_jac * cx;
    //         Eigen::Matrix<double, constraint_dim, control_dim> jac_proj_cu = proj_jac * cu;
    //         dxdx = mu_ * ((jac_proj_cx.transpose() * jac_proj_cx) - std::get<0>(ans));
    //         dxdu = mu_ * ((jac_proj_cx.transpose() * jac_proj_cu) - std::get<2>(ans));
    //         dudu = mu_ * ((jac_proj_cu.transpose() * jac_proj_cu) - std::get<1>(ans));
    //     }
    // }

public:
    Eigen::VectorXd lambda_;
    Eigen::VectorXd lambda_proj_;
    Eigen::VectorXd c_;
    Eigen::MatrixXd cx_;
    std::vector<Eigen::Matrix<double, state_dim, state_dim>> cxx_;
    Eigen::MatrixXd cu_;
    Eigen::MatrixXd proj_cx_T_;
    Eigen::MatrixXd proj_cu_T_;
    Eigen::Matrix<double, state_dim, state_dim> dxdx_;
    Eigen::Matrix<double, control_dim, control_dim> dudu_;
    Eigen::Matrix<double, state_dim, control_dim> dxdu_;


    double mu_;
    bool is_equality_;
    Eigen::MatrixXd proj_jac_;




    // Helper function for tensor contraction
    std::tuple<Eigen::Matrix<double, state_dim, state_dim>, Eigen::Matrix<double, control_dim, control_dim>, Eigen::Matrix<double, state_dim, control_dim>>
    tensor_contract(const Eigen::VectorXd& factor, 
                                    const std::tuple<std::vector<Eigen::Matrix<double, state_dim, state_dim>>,
                       std::vector<Eigen::Matrix<double, control_dim, control_dim>>,
                       std::vector<Eigen::Matrix<double, state_dim, control_dim>>>& tensor) const {

        Eigen::Matrix<double, state_dim, state_dim> factor_dot_hxx;
        Eigen::Matrix<double, control_dim, control_dim> factor_dot_huu;
        Eigen::Matrix<double, state_dim, control_dim> factor_dot_hxu;
        auto hxx = std::get<0>(tensor);
        auto huu = std::get<1>(tensor);
        auto hxu = std::get<2>(tensor);

        factor_dot_hxx.setZero();
        factor_dot_huu.setZero();
        factor_dot_hxu.setZero();


        for(size_t index = 0; index < factor.size(); ++index) {
            factor_dot_hxx += hxx[index] * factor(index, 0);
            factor_dot_huu += huu[index] * factor(index, 0);
            factor_dot_hxu += hxu[index] * factor(index, 0);
        }
        return {factor_dot_hxx, factor_dot_huu, factor_dot_hxu};
    }
};

#endif // CONSTRAINTS_DYNAMIC_CONSTRAINTS_H_
