load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_library(
    name = "al_ilqr",
    srcs = ["al_ilqr.cc"],
    hdrs = [
        "al_ilqr.h", 
        "io_struct.h",
        ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        "@eigen",
        "//model:all_models",
        ],
)

cc_binary(
    name = "al_ilqr_test",
    srcs = ["al_ilqr_test.cc"],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        "//:al_ilqr",
    ],
)

cc_binary(
    name = "test_new_al_ilqr_signal",
    srcs = ["test_new_al_ilqr_signal.cc"],
    deps = [
        "//model:new_bicycle_node",
        "//constraints:box_constraints",
        ":new_al_ilqr",
    ],
    copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)

cc_library(
    name = "new_al_ilqr",
    hdrs = [
        "new_al_ilqr.h", 
        ],
    deps = [
        "//model:new_ilqr_node",
        "//model:new_bicycle_node",
        "//constraints:box_constraints",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
)


cc_binary(
    name = "test_new_al_ilqr",
    srcs = ["test_new_al_ilqr.cc"],
    deps = [
        ":new_al_ilqr",
        "//model:new_bicycle_node",
        "//constraints:box_constraints",
        "//constraints:quadratic_constraints",
    ],
    copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)

cc_binary(
    name = "test_lat_al_ilqr",
    srcs = ["test_lat_al_ilqr.cc"],
    deps = [
        ":new_al_ilqr",
        "//model:new_lat_bicycle_node",
        "//constraints:box_constraints",
        "//constraints:quadratic_constraints",
    ],
    copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)


pybind_extension(
     name = "ilqr_pybind",
     srcs = ["ilqr_pybind.cc"],
     deps = [
         "//constraints:box_constraints",
         "//constraints:constraints",
         "//constraints:linear_constraints",
         "//constraints:constraints_bind",
         "//model:node_bind",
         ":new_al_ilqr",
         "@eigen",
     ],
     copts = ["-O3", "-march=native", "-faligned-new", "-DEIGEN_VECTORIZE"],
)

py_library(
     name = "ilqr_pybind_lib",
     data = [":ilqr_pybind.so"],
     visibility = ["//visibility:public"],
)

