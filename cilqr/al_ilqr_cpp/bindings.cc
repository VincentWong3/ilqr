#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "al_ilqr.h"
#include "io_struct.h"

namespace py = pybind11;

PYBIND11_MODULE(al_ilqr_py, m) {
    py::enum_<ModelType>(m, "ModelType")
        .value("LAT_BICYCLE_KINEMATIC_MODEL", LAT_BICYCLE_KINEMATIC_MODEL)
        .value("LAT_BICYCLE_DYNAMIC_MODEL", LAT_BICYCLE_DYNAMIC_MODEL)
        .value("FULL_BICYCLE_KINEMATIC_MODEL", FULL_BICYCLE_KINEMATIC_MODEL)
        .value("FULL_BICYCLE_DYNAMIC_MODEL", FULL_BICYCLE_DYNAMIC_MODEL);

    py::class_<ALILQRInput>(m, "ALILQRInput")
        .def(py::init<>())
        .def_readwrite("initial_state", &ALILQRInput::initial_state)
        .def_readwrite("A", &ALILQRInput::A)
        .def_readwrite("B", &ALILQRInput::B)
        .def_readwrite("model_type", &ALILQRInput::model_type);

    py::class_<ALILQROutput>(m, "ALILQROutput")
        .def(py::init<>())
        .def_readwrite("optimized_state", &ALILQROutput::optimized_state)
        .def_readwrite("optimized_control", &ALILQROutput::optimized_control);

    py::class_<ALILQR>(m, "ALILQR")
        .def(py::init<int, int>())
        .def("initialize", &ALILQR::initialize)
        .def("optimize", &ALILQR::optimize);
}
