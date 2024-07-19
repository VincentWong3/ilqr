cc_library(
    name = "ilqr_node",
    hdrs = [
        "ilqr_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "full_bicycle_dynamic_node",
    hdrs = [
        "full_bicycle_dynamic_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":ilqr_node",
        "@eigen",  # Ensure Eigen dependency is included here as well
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "full_bicycle_kinematic_node",
    hdrs = [
        "full_bicycle_kinematic_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":ilqr_node",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "all_models",
    hdrs = [
        "ilqr_node.h",
        "full_bicycle_dynamic_node.h",
        "full_bicycle_kinematic_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":ilqr_node",
        ":full_bicycle_dynamic_node",
        ":full_bicycle_kinematic_node",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fast_ilqr_node",
    hdrs = [
        "fast_ilqr_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "fast_full_bicycle_dynamic_node",
    srcs = ["fast_full_bicycle_dynamic_node.cc"],
    hdrs = [
        "fast_full_bicycle_dynamic_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":fast_ilqr_node",
        "//:fast_sine",
    ],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "test_models",
    srcs = [
        "test_models.cc",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":fast_full_bicycle_dynamic_node",
        ":all_models",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)