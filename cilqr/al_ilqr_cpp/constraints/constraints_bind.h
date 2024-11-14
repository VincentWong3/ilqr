#ifndef CONSTRAINTS_BIND_H_
#define CONSTRAINTS_BIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <array>
#include "constraints.h"
#include "linear_constraints.h"
#include "box_constraints.h"
#include "quadratic_constraints.h"

namespace py = pybind11;

template <int state_dim, int control_dim, int constraint_dim>
void bind_constraints(py::module& m, const std::string& class_name) {
    using ConstraintsType = Constraints<state_dim, control_dim, constraint_dim>;

    py::class_<ConstraintsType>(m, class_name.c_str())
        // 无需绑定构造函数
        .def_property("lambda", &ConstraintsType::lambda, &ConstraintsType::set_lambda)
        .def_property("mu", &ConstraintsType::mu, &ConstraintsType::set_mu)
        .def_static("get_state_dim", &ConstraintsType::get_state_dim)
        .def_static("get_control_dim", &ConstraintsType::get_control_dim)
        .def_static("get_constraint_dim", &ConstraintsType::get_constraint_dim)
        .def("projection", &ConstraintsType::projection)
        .def("projection_jacobian", &ConstraintsType::projection_jacobian)
        .def("update_mu", &ConstraintsType::update_mu)
        ;
}

template <int state_dim, int control_dim, int constraint_dim>
void bind_linear_constraints(py::module& m, const std::string& class_name) {
    using ConstraintsType = Constraints<state_dim, control_dim, constraint_dim>;
    using LinearConstraintsType = LinearConstraints<state_dim, control_dim, constraint_dim>;

    py::class_<LinearConstraintsType, ConstraintsType>(m, class_name.c_str())
        // 绑定构造函数
        .def(py::init<
            const Eigen::Matrix<double, constraint_dim, state_dim>&,
            const Eigen::Matrix<double, constraint_dim, control_dim>&,
            const Eigen::Matrix<double, constraint_dim, 1>&,
            bool>(),
            py::arg("A"), py::arg("B"), py::arg("C"), py::arg("is_equality") = false)
        // 绑定成员函数
        .def("constraints", &LinearConstraintsType::constraints)
        .def("parallel_constraints", &LinearConstraintsType::parallel_constraints)
        .def("constraints_jacobian", &LinearConstraintsType::constraints_jacobian)
        .def("constraints_hessian", &LinearConstraintsType::constraints_hessian)
        ;
}

template <int state_dim, int control_dim>
void bind_box_constraints(py::module& m, const std::string& class_name) {
    constexpr int constraint_dim = 2 * (state_dim + control_dim);

    using BaseConstraintsType = LinearConstraints<state_dim, control_dim, constraint_dim>;
    using BoxConstraintsType = BoxConstraints<state_dim, control_dim>;

    py::class_<BoxConstraintsType, BaseConstraintsType>(m, class_name.c_str())
        .def(py::init<
            const Eigen::Matrix<double, state_dim, 1>&,
            const Eigen::Matrix<double, state_dim, 1>&,
            const Eigen::Matrix<double, control_dim, 1>&,
            const Eigen::Matrix<double, control_dim, 1>&>(),
            py::arg("state_min"), py::arg("state_max"),
            py::arg("control_min"), py::arg("control_max")
        )
        ;
}

template <int state_dim, int control_dim, int constraint_dim>
void bind_quadratic_constraints(py::module& m, const std::string& class_name) {
    using ConstraintsType = Constraints<state_dim, control_dim, constraint_dim>;
    using QuadraticConstraintsType = QuadraticConstraints<state_dim, control_dim, constraint_dim>;

    py::class_<QuadraticConstraintsType, ConstraintsType>(m, class_name.c_str())
        // 绑定构造函数
        .def(py::init<
            const std::array<Eigen::Matrix<double, state_dim, state_dim>, constraint_dim>&,
            const Eigen::Matrix<double, constraint_dim, state_dim>&,
            const Eigen::Matrix<double, constraint_dim, control_dim>&,
            const Eigen::Matrix<double, constraint_dim, 1>&,
            bool>(),
            py::arg("Q"), py::arg("A"), py::arg("B"), py::arg("C"), py::arg("is_equality") = false)
        // 绑定成员函数
        .def("constraints", &QuadraticConstraintsType::constraints)
        .def("parallel_constraints", &QuadraticConstraintsType::parallel_constraints)
        .def("constraints_jacobian", &QuadraticConstraintsType::constraints_jacobian)
        .def("constraints_hessian", &QuadraticConstraintsType::constraints_hessian)
        ;
}

#endif  // CONSTRAINTS_BIND_H_
