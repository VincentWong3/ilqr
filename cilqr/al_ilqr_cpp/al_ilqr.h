#ifndef ALILQR_H
#define ALILQR_H

#include <vector>
#include <Eigen/Dense>
#include <tuple>
#include "model/full_bicycle_dynamic_node.h"
#include "model/full_bicycle_kinematic_node.h"

class ALILQR {
public:
    ALILQR(const std::vector<ILQRNode*>& ilqr_nodes);
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> linearizedInitialGuess();
    double computeTotalCost();
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> backward();
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> forward(const Eigen::MatrixXd& k, const Eigen::MatrixXd& K);
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> optimize(int max_iters = 20, double tol = 1e-8);
    void updateLambda();
    void updateMu(double gain);
    double computeConstraintViolation();

private:
    std::vector<ILQRNode*> ilqr_nodes;
    int horizon;
};

#endif // ALILQR_H
