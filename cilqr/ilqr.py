import numpy as np
import matplotlib.pyplot as plt
from lat_bicycle_node import LatBicycleKinematicNode


class ILQR:
    def __init__(self, ilqr_nodes):
        self.ilqr_nodes = ilqr_nodes
        self.horizon = len(ilqr_nodes) - 1

    def linearized_initial_guess(self):
        horizon = self.horizon
        state_dim = self.ilqr_nodes[0].state_dim
        control_dim = self.ilqr_nodes[0].control_dim

        x = np.zeros((horizon + 1, state_dim))
        u = np.zeros((horizon, control_dim))

        x[0] = self.ilqr_nodes[0].state  # 初始状态

        P = self.ilqr_nodes[-1].Q
        K_list = []

        for t in reversed(range(horizon)):
            current_node = self.ilqr_nodes[t]
            A, B = current_node.dynamics_jacobian(current_node.goal, np.zeros(control_dim))
            K = np.linalg.inv(current_node.R + B.T @ P @ B) @ (B.T @ P @ A)
            K_list.append(K)
            P = current_node.Q + A.T @ P @ (A - B @ K)

        K_list = K_list[::-1]

        for t in range(horizon):
            current_node = self.ilqr_nodes[t]
            goal_state = current_node.goal

            K = K_list[t]
            u[t] = -K @ (x[t] - goal_state)
            u[t] = np.clip(u[t], current_node.control_min, current_node.control_max)

            x[t + 1] = current_node.dynamics(x[t], u[t])

            self.ilqr_nodes[t].control = u[t]
            self.ilqr_nodes[t + 1].state = x[t + 1]

        for node in self.ilqr_nodes:
            node.set_lambda(np.zeros(node.constraint_dim))
            node.set_mu(1.0)

        return x, u

    def compute_total_cost(self):
        total_cost = 0
        for node in self.ilqr_nodes:
            total_cost += node.cost()
        return total_cost

    def backward(self):
        horizon = self.horizon
        state_dim = self.ilqr_nodes[0].state_dim
        control_dim = self.ilqr_nodes[0].control_dim

        A = np.zeros((horizon, state_dim, state_dim))
        B = np.zeros((horizon, state_dim, control_dim))
        cost_Jx = np.zeros((horizon + 1, state_dim))
        cost_Ju = np.zeros((horizon, control_dim))
        cost_Hx = np.zeros((horizon + 1, state_dim, state_dim))
        cost_Hu = np.zeros((horizon, control_dim, control_dim))

        for t in range(horizon):
            node = self.ilqr_nodes[t]
            A[t], B[t] = node.dynamics_jacobian(node.state, node.control)
            cost_Jx[t], cost_Ju[t] = node.cost_jacobian()
            cost_Hx[t], cost_Hu[t] = node.cost_hessian()

        cost_Jx[-1], _ = self.ilqr_nodes[-1].cost_jacobian()
        cost_Hx[-1], _ = self.ilqr_nodes[-1].cost_hessian()

        Vx = cost_Jx[-1]
        Vxx = cost_Hx[-1]

        K = np.zeros((horizon, control_dim, state_dim))
        k = np.zeros((horizon, control_dim))

        for t in reversed(range(horizon)):
            Qu = cost_Ju[t] + B[t].T @ Vx
            Qx = cost_Jx[t] + A[t].T @ Vx
            Qux = B[t].T @ Vxx @ A[t]
            Quu = cost_Hu[t] + B[t].T @ Vxx @ B[t]
            Qxx = cost_Hx[t] + A[t].T @ Vxx @ A[t]

            Quu_inv = np.linalg.inv(Quu + np.eye(control_dim) * 1e-9)  # Regularization

            K[t] = -Quu_inv @ Qux
            k[t] = -Quu_inv @ Qu

            Vx = Qx + K[t].T @ Quu @ k[t] + K[t].T @ Qu + Qux.T @ k[t]
            Vxx = Qxx + K[t].T @ Quu @ K[t] + K[t].T @ Qux + Qux.T @ K[t]

        return k, K

    def forward(self, k, K):
        alpha = 1.0
        horizon = self.horizon
        state_dim = self.ilqr_nodes[0].state_dim
        control_dim = self.ilqr_nodes[0].control_dim

        x = np.array([node.state for node in self.ilqr_nodes])
        u = np.array([node.control for node in self.ilqr_nodes[:-1]])

        new_x = np.zeros_like(x)
        new_u = np.zeros_like(u)

        new_x[0] = x[0]
        old_cost = self.compute_total_cost()

        while alpha > 1e-8:
            for t in range(horizon):
                new_u[t] = u[t] + alpha * k[t] + K[t] @ (new_x[t] - x[t])
                new_x[t + 1] = self.ilqr_nodes[t].dynamics(new_x[t], new_u[t])
                self.ilqr_nodes[t + 1].state = new_x[t + 1]
                self.ilqr_nodes[t].control = new_u[t]
            new_cost = self.compute_total_cost()
            if new_cost < old_cost:
                break
            else:
                alpha = alpha / 2.0
        if alpha <= 1e-8:
            new_x = x
            new_u = u
            for t in range(horizon):
                self.ilqr_nodes[t + 1].state = new_x[t + 1]
                self.ilqr_nodes[t].control = new_u[t]

        return new_x, new_u

    def optimize(self, max_iters=20, tol=1e-8):
        x_init, u_init = self.linearized_initial_guess()
        old_cost = self.compute_total_cost()
        x, u = x_init, u_init
        for j in range(10):
            old_cost = self.compute_total_cost()
            for i in range(max_iters):
                k, K = self.backward()
                new_x, new_u = self.forward(k, K)
                new_cost = self.compute_total_cost()
                if abs(new_cost - old_cost) < tol:
                    break
                x, u = new_x, new_u
                old_cost = new_cost
            violation = self.compute_constrain_violation()

            if violation < 1e-2:
                break
            elif violation >= 1e-2 and violation < 1e-1:
                self.update_lambda()
            else:
                self.update_mu(4.0)
        return x_init, u_init, x, u


    def update_lambda(self):
        for node in self.ilqr_nodes:
            node.update_lambda()

    def update_mu(self, gain):
        for node in self.ilqr_nodes:
            node.set_mu(node.mu * gain)

    def compute_constrain_violation(self):
        violation = 0.0
        for node in self.ilqr_nodes:
            one_constrain = np.maximum(-node.lambda_ / node.mu, node.constraints())
            one_violation = one_constrain.T @ one_constrain
            violation += np.sqrt(one_violation)
        return violation

