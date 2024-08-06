cc_library(
    name = "constraints",
    hdrs = [
        "constraints.h",
    ],
    copts = ["-O3", "-march=native"],
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
    copts = ["-O3", "-march=native"],
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
    copts = ["-O3", "-march=native"],
    deps = [
        ":linear_constraints",
    ],
    visibility = ["//visibility:public"],
)


cc_binary(
    name = "test_constraints",
    srcs = [
        "test_constraints.cc",
    ],
    copts = ["-O3", "-march=native"],
    deps = [
        ":constraints",
        ":box_constraints",
        ":linear_constraints",
    ],
    visibility = ["//visibility:public"],
)