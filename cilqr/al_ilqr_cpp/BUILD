load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

cc_library(
    name = "al_ilqr",
    srcs = ["al_ilqr.cc"],
    hdrs = [
        "al_ilqr.h", 
        "io_struct.h",
        ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
        "//model:all_models",
        ],
)

cc_binary(
    name = "al_ilqr_test",
    srcs = ["al_ilqr_test.cc"],
    copts = ["-O3", "-march=native"],
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
    copts = ["-O3", "-march=native", "-faligned-new"],
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
    copts = ["-O3", "-march=native"],
)



cc_binary(
    name = "test_new_al_ilqr",
    srcs = ["test_new_al_ilqr.cc"],
    deps = [
        ":new_al_ilqr",
        "//model:new_bicycle_node",
        "//constraints:box_constraints",
    ],
    copts = ["-O3", "-march=native", "-faligned-new"],
)



# pybind_extension(
#     name = "al_ilqr_py",
#     srcs = ["bindings.cc"],
#     deps = [
#         ":al_ilqr",
#         "@eigen",
#     ],
# )

# py_library(
#     name = "al_ilqr_py",
#     data = [":al_ilqr_py.so"],
#     visibility = ["//visibility:public"],
# )

