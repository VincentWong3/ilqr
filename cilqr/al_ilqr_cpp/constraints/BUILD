load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_library(
    name = "constraints",
    hdrs = [
        "constraints.h",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "linear_constraints",
    hdrs = [
        "linear_constraints.h",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        ":constraints",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "box_constraints",
    hdrs = [
        "box_constraints.h",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        ":linear_constraints",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "quadratic_constraints",
    hdrs = [
        "quadratic_constraints.h",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        ":constraints",
    ],
    visibility = ["//visibility:public"],
)


cc_binary(
    name = "test_constraints",
    srcs = [
        "test_constraints.cc",
    ],
    deps = [
        ":constraints",
        ":box_constraints",
        ":linear_constraints",
        ":quadratic_constraints",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "constraints_bind",
    hdrs = [
        "constraints_bind.h",
    ],
    copts = ["-O3", "-march=native", "-DEIGEN_VECTORIZE"],
    deps = [
        ":constraints",
        ":box_constraints",
        ":linear_constraints",
        ":quadratic_constraints",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)


#pybind_extension(
#    name = "constraints_py",
#    srcs = [
#        "constraints_bind.cc",
#        "constraints_bind.h",
#    ],
#    deps = [
#        "//constraints:box_constraints",
#        "//constraints:constraints",
#        "//constraints:linear_constraints",
#        "@eigen",
#    ],
#)

#py_library(
#    name = "constraints_py_lib",
#    data = [":constraints_py.so"],
#    visibility = ["//visibility:public"],
#)