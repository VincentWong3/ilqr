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

template <int state_dim, int control_dim>
void bind_new_al_ilqr(py::module& m, const std::string& class_name) {
    using NewALILQRType = NewALILQR<state_dim, control_dim>;
    using VectorState = Eigen::Matrix<double, state_dim, 1>;
    using ILQRNodeType = NewILQRNode<state_dim, control_dim>;

    using ILQRNodeVector = std::vector<std::shared_ptr<ILQRNodeType>>;

    using ObsVector = std::vector<Eigen::Matrix<double, 2, 4>>;

    py::class_<NewALILQRType, std::shared_ptr<NewALILQRType>>(m, class_name.c_str())
        .def(py::init<const ILQRNodeVector&, const VectorState&>(),
             py::arg("ilqr_nodes"), py::arg("init_state"))
        .def(py::init<const ILQRNodeVector&, const VectorState&, const ObsVector&, const ObsVector&>(),
             py::arg("ilqr_nodes"), py::arg("init_state"), py::arg("left_obs"), py::arg("right_obs"))
        .def("optimize", &NewALILQRType::optimize,
             py::arg("max_outer_iter"), py::arg("max_inner_iter"), py::arg("max_violation"))
        .def("get_x_list", &NewALILQRType::get_x_list)
        .def("get_u_list", &NewALILQRType::get_u_list);
}

PYBIND11_MODULE(ilqr_pybind, m) {
    
    bind_constraints<6, 2, 16>(m, "Constraints6_2_16");

    bind_constraints<6, 2, 5>(m, "Constraints6_2_5");

    bind_constraints<4, 1, 10>(m, "Constraints4_1_10");

    bind_constraints<4, 1, 6>(m, "Constraints4_1_6");

    bind_constraints<4, 1, 3>(m, "Constraints4_1_3");

    bind_linear_constraints<6, 2, 16>(m, "LinearConstraints6_2_16");

    bind_linear_constraints<4, 1, 10>(m, "LinearConstraints4_1_10");

    bind_linear_constraints<4, 1, 6>(m, "LinearConstraints4_1_6");

    bind_box_constraints<6, 2>(m, "BoxConstraints6_2");

    bind_box_constraints<4, 1>(m, "BoxConstraints4_1");

    bind_quadratic_constraints<6, 2, 5>(m, "QuadraticConstraints6_2_5");

    bind_quadratic_constraints<4, 1, 3>(m, "QuadraticConstraints4_1_3");

    bind_new_ilqr_node<6, 2>(m, "ILQRNode6_2");

    bind_new_ilqr_node<4, 1>(m, "ILQRNode4_1");

    bind_new_lat_bicycle_node_inner<LinearConstraints<4, 1, 10>>(m, "NewLatBicycleNodeLinearConstraints4_1_10");

    bind_new_bicycle_node<BoxConstraints<6, 2>>(m, "NewBicycleNodeBoxConstraints6_2");

    bind_new_bicycle_node<QuadraticConstraints<6, 2, 5>>(m, "NewBicycleNodeQuadraticConstraints6_2_5");

    bind_new_lat_bicycle_node<QuadraticConstraints<4, 1, 3>>(m, "NewLatBicycleNodeQuadraticConstraints4_1_3");

    bind_new_lat_bicycle_node<LinearConstraints<4, 1, 6>>(m, "NewLatBicycleNodeLinearConstraints4_1_6");


    bind_new_al_ilqr<6, 2>(m, "NewALILQR6_2");

    bind_new_al_ilqr<4, 1>(m, "NewALILQR4_1");
}




