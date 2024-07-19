#ifndef IO_STRUCT_H
#define IO_STRUCT_H

#include <Eigen/Dense>

enum ModelType {
    LAT_BICYCLE_KINEMATIC_MODEL,
    LAT_BICYCLE_DYNAMIC_MODEL,
    FULL_BICYCLE_KINEMATIC_MODEL,
    FULL_BICYCLE_DYNAMIC_MODEL
};

struct ALILQRInput {
    Eigen::VectorXd initial_state;
    Eigen::MatrixXd A;
    Eigen::MatrixXd B;
    ModelType model_type;
};

struct ALILQROutput {
    Eigen::VectorXd optimized_state;
    Eigen::VectorXd optimized_control;
};

#endif // IO_STRUCT_H
