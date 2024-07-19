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

cc_library(
    name = "fast_sine",
    srcs = ["fast_sine.cc"],
    hdrs = ["fast_sine.h"],
    copts = ["-O3", "-march=native"],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "no_alias",
    srcs = ["test_no_alias.cc"],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
        ],
)

cc_binary(
    name = "test_fast_sine",
    srcs = ["test_fast_sine.cc"],
    deps = [
        ":fast_sine",
    ],
    copts = ["-O3", "-march=native"],
)

cc_library(
    name = "fast_al_ilqr",
    hdrs = [
        "fast_al_ilqr.h", 
        ],
    copts = ["-O2", "-march=native"],
    deps = [
        "@eigen",
        "//model:all_models",
        "//model:fast_full_bicycle_dynamic_node",
        ],
)

cc_binary(
    name = "test_fast_al_ilqr",
    srcs = ["test_fast_al_ilqr.cc"],
    deps = [
        ":fast_sine",
        ":fast_al_ilqr",
    ],
    copts = ["-O2", "-march=native"],
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

