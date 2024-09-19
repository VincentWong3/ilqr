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
        "@eigen",
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
    name = "new_ilqr_node",
    hdrs = [
        "new_ilqr_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
        "//constraints:box_constraints",
        ":parallel_compution_function",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "new_bicycle_node",
    hdrs = [
        "new_bicycle_node.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
        "//constraints:box_constraints",
        ":new_ilqr_node",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "parallel_compution_function",
    hdrs = [
        "parallel_compution_function.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "node_bind",
    hdrs = [
        "node_bind.h",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":new_bicycle_node",
        "@eigen",
    ],
    visibility = ["//visibility:public"],
)

