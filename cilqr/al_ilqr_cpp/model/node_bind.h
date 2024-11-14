#ifndef NEW_ILQR_NODE_BIND_H_
#define NEW_ILQR_NODE_BIND_H_

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "new_ilqr_node.h"
#include "constraints/constraints.h"
#include "constraints/linear_constraints.h"
#include "constraints/box_constraints.h"
#include "model/new_bicycle_node.h"
#include "model/new_lat_bicycle_node.h"

namespace py = pybind11;

template <int state_dim, int control_dim>
void bind_new_ilqr_node(py::module& m, const std::string& class_name) {
    using NewILQRNodeType = NewILQRNode<state_dim, control_dim>;

    py::class_<NewILQRNodeType, std::shared_ptr<NewILQRNodeType>>(m, class_name.c_str())
        // 不要绑定抽象类的构造函数
        // .def(py::init<const VectorState&>(), py::arg("goal"))  // 已删除
        .def_static("get_state_dim", &NewILQRNodeType::get_state_dim)
        .def_static("get_control_dim", &NewILQRNodeType::get_control_dim)
        .def("normalize_angle", &NewILQRNodeType::normalize_angle)
        .def("goal", &NewILQRNodeType::goal)
        // 绑定纯虚函数
        .def("dynamics", &NewILQRNodeType::dynamics)
        .def("cost", &NewILQRNodeType::cost)
        .def("parallel_cost", &NewILQRNodeType::parallel_cost)
        .def("dynamics_jacobian", &NewILQRNodeType::dynamics_jacobian)
        .def("dynamics_hessian_fxx", &NewILQRNodeType::dynamics_hessian_fxx)
        .def("cost_jacobian", &NewILQRNodeType::cost_jacobian)
        .def("cost_hessian", &NewILQRNodeType::cost_hessian)
        .def("update_lambda", &NewILQRNodeType::update_lambda)
        .def("update_mu", &NewILQRNodeType::update_mu)
        .def("reset_lambda", &NewILQRNodeType::reset_lambda)
        .def("reset_mu", &NewILQRNodeType::reset_mu)
        .def("max_constraints_violation", &NewILQRNodeType::max_constraints_violation)
        ;
}


template <typename ConstraintsType>
void bind_new_bicycle_node(py::module& m, const std::string& class_name) {
    using NewBicycleNodeType = NewBicycleNode<ConstraintsType>;
    using BaseClass = NewILQRNode<6, 2>;
    using VectorState = Eigen::Matrix<double, 6, 1>;
    using MatrixQ = Eigen::Matrix<double, 6, 6>;
    using MatrixR = Eigen::Matrix<double, 2, 2>;

    py::class_<NewBicycleNodeType, BaseClass, std::shared_ptr<NewBicycleNodeType>>(m, class_name.c_str())
        .def(py::init<double, double, double, const VectorState&, const MatrixQ&, const MatrixR&, const ConstraintsType&>(),
             py::arg("L"), py::arg("dt"), py::arg("k"), py::arg("goal"), py::arg("Q"), py::arg("R"), py::arg("constraints"))
        .def("dynamics", &NewBicycleNodeType::dynamics)
        .def("cost", &NewBicycleNodeType::cost)
        .def("parallel_cost", &NewBicycleNodeType::parallel_cost)
        .def("dynamics_jacobian", &NewBicycleNodeType::dynamics_jacobian)
        .def("dynamics_hessian_fxx", &NewBicycleNodeType::dynamics_hessian_fxx)
        .def("cost_jacobian", &NewBicycleNodeType::cost_jacobian)
        .def("cost_hessian", &NewBicycleNodeType::cost_hessian)
        .def("update_lambda", &NewBicycleNodeType::update_lambda)
        .def("update_mu", &NewBicycleNodeType::update_mu)
        .def("reset_lambda", &NewBicycleNodeType::reset_lambda)
        .def("reset_mu", &NewBicycleNodeType::reset_mu)
        .def("max_constraints_violation", &NewBicycleNodeType::max_constraints_violation)
        ;
}

template <typename ConstraintsType>
void bind_new_lat_bicycle_node(py::module& m, const std::string& class_name) {
    using NewLatBicycleNodeType = NewLatBicycleNode<ConstraintsType>;
    using BaseClass = NewILQRNode<4, 1>;
    using VectorState = Eigen::Matrix<double, 4, 1>;
    using MatrixQ = Eigen::Matrix<double, 4, 4>;
    using MatrixR = Eigen::Matrix<double, 1, 1>;

    py::class_<NewLatBicycleNodeType, BaseClass, std::shared_ptr<NewLatBicycleNodeType>>(m, class_name.c_str())
        .def(py::init<double, double, double, double, const VectorState&, const MatrixQ&, const MatrixR&, const ConstraintsType&>(),
             py::arg("L"), py::arg("dt"), py::arg("k"), py::arg("v"), py::arg("goal"), py::arg("Q"), py::arg("R"), py::arg("constraints"))
        .def("dynamics", &NewLatBicycleNodeType::dynamics)
        .def("cost", &NewLatBicycleNodeType::cost)
        .def("parallel_cost", &NewLatBicycleNodeType::parallel_cost)
        .def("dynamics_jacobian", &NewLatBicycleNodeType::dynamics_jacobian)
        .def("dynamics_hessian_fxx", &NewLatBicycleNodeType::dynamics_hessian_fxx)
        .def("cost_jacobian", &NewLatBicycleNodeType::cost_jacobian)
        .def("cost_hessian", &NewLatBicycleNodeType::cost_hessian)
        .def("update_lambda", &NewLatBicycleNodeType::update_lambda)
        .def("update_mu", &NewLatBicycleNodeType::update_mu)
        .def("reset_lambda", &NewLatBicycleNodeType::reset_lambda)
        .def("reset_mu", &NewLatBicycleNodeType::reset_mu)
        .def("max_constraints_violation", &NewLatBicycleNodeType::max_constraints_violation)
        ;
}


#endif // NEW_ILQR_NODE_BIND_H_