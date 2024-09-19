#include <iostream>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include "constraints/constraints_bind.h"
#include "model/node_bind.h"
#include "new_al_ilqr.h"


namespace py = pybind11;

template <int state_dim, int control_dim, int horizon>
void bind_new_al_ilqr(py::module& m, const std::string& class_name) {
    using NewALILQRType = NewALILQR<state_dim, control_dim>;
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using ILQRNodeType = NewILQRNode<state_dim, control_dim>;

    using ILQRNodeVector = std::vector<std::shared_ptr<ILQRNodeType>>;

    py::class_<NewALILQRType, std::shared_ptr<NewALILQRType>>(m, class_name.c_str())
        .def(py::init<const ILQRNodeVector&, const VectorState&>(),
             py::arg("ilqr_nodes"), py::arg("init_state"))
        .def("optimize", &NewALILQRType::optimize,
             py::arg("max_outer_iter"), py::arg("max_inner_iter"), py::arg("max_violation"))
        .def("get_x_list", &NewALILQRType::get_x_list)
        .def("get_u_list", &NewALILQRType::get_u_list);
}

PYBIND11_MODULE(ilqr_pybind, m) {
    constexpr int state_dim = 6;
    constexpr int control_dim = 2;
    constexpr int constraint_dim = 2 * (state_dim + control_dim);
    constexpr int parallel_num = PARALLEL_NUM;  // 确保与定义一致

    // 绑定 Constraints<6, 2, 16> 类
    bind_constraints<state_dim, control_dim, constraint_dim>(m, "Constraints6_2_16");

    // 绑定 LinearConstraints<6, 2, 16> 类
    bind_linear_constraints<state_dim, control_dim, constraint_dim, parallel_num>(m, "LinearConstraints6_2_16");

    // 绑定 BoxConstraints<6, 2> 类
    bind_box_constraints<state_dim, control_dim>(m, "BoxConstraints6_2");

    bind_new_ilqr_node<state_dim, control_dim>(m, "ILQRNode6_2");

    bind_new_bicycle_node<BoxConstraints<state_dim, control_dim>>(m, "NewBicycleNode6_2");

    bind_new_al_ilqr<state_dim, control_dim, 50>(m, "NewALILQR6_2");
}




