import numpy as np
from fast_ilqr_node import *
from constraints import *
from box_constrains import *
from full_bicycle_dynamic_node import *

class FastBicycleNode(FastILQRNode):
    def __init__(self, L, dt, k, goal, Q, R, constraints):
        state_dim = 6
        control_dim = 2
        super().__init__(state_dim, control_dim, goal, constraints)
        self.L = L
        self.dt = dt
        self.k = k
        self.Q = np.array(Q)
        self.R = np.array(R)

    def normalize_angle(self, angle):
        """Normalize angle to be within the interval (-π, π)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def normalize_state(self):
        """Normalize the angles in the state."""
        self.state[2] = self.normalize_angle(self.state[2])  # Normalize theta
        self.state[3] = self.normalize_angle(self.state[3])  # Normalize delta

    def dynamics(self, state, control):
        x, y, theta, delta, v, a = state
        u1, u2 = control
        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)

        # Runge-Kutta 2nd Order (RK2) method for discretization
        k1 = self.dynamics_continuous(state, control)
        mid_state = state + 0.5 * self.dt * k1
        k2 = self.dynamics_continuous(mid_state, control)

        state_next = state + self.dt * k2
        state_next[2] = self.normalize_angle(state_next[2])  # Normalize theta
        state_next[3] = self.normalize_angle(state_next[3])  # Normalize delta
        return state_next

    def dynamics_continuous(self, state, control):
        x, y, theta, delta, v, a = state.flatten()
        u1, u2 = control.flatten()
        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = v * np.tan(delta) / (self.L * (1 + self.k * v**2))
        delta_dot = u1
        v_dot = a
        a_dot = u2
        return np.array([x_dot, y_dot, theta_dot, delta_dot, v_dot, a_dot])

    def dynamics_jacobian(self, state=None, control=None):
        if state is None:
            state = self.state
        if control is None:
            control = self.control

        x, y, theta, delta, v, a = state.flatten()
        u1, u2 = control.flatten()

        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)
        dt = self.dt
        L = self.L
        k = self.k

        theta_mid = theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))
        v_term = 0.5 * a * dt + v
        tan_delta = np.tan(delta)
        tan_delta_mid = np.tan(delta + 0.5 * dt * u1)
        k_v_sq = k * v ** 2
        k_v_mid_sq = k * v_term ** 2
        denom = L * (k_v_sq + 1)
        denom_mid = L * (k_v_mid_sq + 1)
        cos_theta_mid = np.cos(theta_mid)
        sin_theta_mid = np.sin(theta_mid)

        # 定义矩阵每一行
        row1 = [
            1,
            0,
            -dt * v_term * sin_theta_mid,
            -0.5 * dt ** 2 * v * v_term * (tan_delta ** 2 + 1) * sin_theta_mid / denom,
            -dt * v_term * (
                        -dt * k_v_sq * tan_delta / denom ** 2 + 0.5 * dt * tan_delta / denom) * sin_theta_mid + dt * cos_theta_mid,
            0.5 * dt ** 2 * cos_theta_mid
        ]

        row2 = [
            0,
            1,
            dt * v_term * cos_theta_mid,
            0.5 * dt ** 2 * v * v_term * (tan_delta ** 2 + 1) * cos_theta_mid / denom,
            dt * v_term * (
                        -dt * k_v_sq * tan_delta / denom ** 2 + 0.5 * dt * tan_delta / denom) * cos_theta_mid + dt * sin_theta_mid,
            0.5 * dt ** 2 * sin_theta_mid
        ]

        row3 = [
            0,
            0,
            1,
            dt * v_term * (tan_delta_mid ** 2 + 1) / denom_mid,
            -dt * k_v_mid_sq * (1.0 * a * dt + 2 * v) * tan_delta_mid / denom_mid ** 2 + dt * tan_delta_mid / denom_mid,
            -dt ** 2 * k_v_mid_sq * tan_delta_mid / denom_mid ** 2 + 0.5 * dt ** 2 * tan_delta_mid / denom_mid
        ]

        row4 = [0, 0, 0, 1, 0, 0]
        row5 = [0, 0, 0, 0, 1, dt]
        row6 = [0, 0, 0, 0, 0, 1]

        # 组装矩阵
        J_discrete = np.array([row1, row2, row3, row4, row5, row6])

        Jx = J_discrete

        # 定义矩阵每一行
        row7 = [0, 0]
        row8 = [0, 0]

        row9 = [
            0.5 * dt ** 2 * v_term * (tan_delta_mid ** 2 + 1) / denom_mid,
            0
        ]

        row10 = [dt, 0]
        row11 = [0, 0.5 * dt ** 2]
        row12 = [0, dt]

        # 组装矩阵
        J_control = np.array([row7, row8, row9, row10, row11, row12])
        Ju = J_control


        return Jx, Ju

    def dynamics_hessian(self, Vx, state=None, control=None):
        if state is None:
            state = self.state
        if control is None:
            control = self.control

        x, y, theta, delta, v, a = state.flatten()
        u1, u2 = control.flatten()

        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)
        dt = self.dt
        L = self.L
        k = self.k
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        tan_delta = np.tan(delta)
        tan_delta_square_plus_one = tan_delta * tan_delta + 1
        k_v_square = k * v * v
        k_v_square_plus_one = k_v_square + 1
        H2 = np.zeros((6, 6))
        H1 = H2
        H0 = H1

        H2[3, 3] = 2 * dt * v * tan_delta_square_plus_one * tan_delta / (k_v_square_plus_one * L)
        H2[3, 4] = dt * (1 - k * v * v) * tan_delta_square_plus_one / (k_v_square_plus_one * L) / (k_v_square_plus_one)
        H2[4, 3] = H2[3, 4]
        H2[4, 4] = dt * 2 * k * v * (k_v_square - 3) * tan_delta / L / k_v_square_plus_one / k_v_square_plus_one / k_v_square_plus_one

        H1[2, 2] = -dt * v * sin_theta
        H1[2, 4] = dt * cos_theta
        H1[4, 2] = H1[2, 4]

        H0[2, 2] = -dt * v * cos_theta
        H0[2, 4] = -dt * sin_theta
        H0[4, 2] = -dt * sin_theta

        result = np.zeros((6, 6))
        result = H0 * Vx[0] + H1 * Vx[1] + H2 * Vx[2]

        return result

    def cost(self):
        self.normalize_state()
        # Define the quadratic cost function
        state_error = self.state - self.goal
        state_cost = state_error.T @ self.Q @ state_error
        control_cost = self.control.T @ self.R @ self.control
        constraints_cost = self.constraints_obj.augmented_lagrangian_cost(self.state, self.control)

        return state_cost + control_cost + constraints_cost

    def cost_jacobian(self):
        self.normalize_state()
        # Compute the Jacobian of the cost function
        state_error = self.state - self.goal
        Jx = 2 * self.Q @ state_error
        Ju = 2 * self.R @ self.control
        constrain_jx, constrain_ju = self.constraints_obj.augmented_lagrangian_jacobian(self.state, self.control)
        Jx = Jx + constrain_jx
        Ju = Ju + constrain_ju

        return Jx, Ju

    def cost_hessian(self):
        # Compute the Hessian of the cost function
        Hx = 2 * self.Q.astype(np.float64)
        Hu = 2 * self.R.astype(np.float64)
        constraint_hx, constraint_hu, _ = self.constraints_obj.augmented_lagrangian_hessian(self.state, self.control)
        Hx += constraint_hx.astype(np.float64)
        Hu += constraint_hu.astype(np.float64)
        return Hx, Hu

# Q_full = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6])
# R_full = np.array([[10, 0], [0, 10]])
#
# state_min = np.array([-1000, -1000, -2 * np.pi, -10, -100, -10])
# state_max = np.array([1000, 1000, 2 * np.pi, 10, 100, 10])
# control_min = np.array([-0.2, -1])
# control_max = np.array([0.2, 1])
#
# goal = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#
# constraints = BoxConstraint(state_min, state_max, control_min, control_max)
#
# x = np.array([0, 0, 0, 0, 10, 0])
# u = np.array([-2, -2])
#
# L = 3.0
# dt = 0.1
# k = 0.0001
#
# node = FastBicycleNode(L, dt, k, goal, Q_full, R_full, constraints)
#
# node.state = x
# node.control = u
#
# x_next = node.dynamics(x, u)
#
# jx, ju = node.dynamics_jacobian(x, u)
#
# cost = node.cost()
#
# c_jx, c_ju = node.cost_jacobian()
#
# c_hx, c_hu = node.cost_hessian()
#
# state_bounds_full = np.array([[-1000, -1000, -2 * np.pi, -10, -100, -10], [1000, 1000, 2 * np.pi, 10, 100, 10]])
# control_bounds_full = np.array([[-0.2, -1], [0.2, 1]])
#
# node2 = FullBicycleDynamicNode(L, dt, k, state_bounds_full, control_bounds_full, goal, Q_full, R_full)
#
# node2.state = x
# node2.control = u
#
# x_next2 = node2.dynamics(x, u)
#
# jx2, ju2 = node2.dynamics_jacobian(x, u)
#
# cost2 = node2.cost()
#
# c_jx2, c_ju2 = node2.cost_jacobian()
#
# c_hx2, c_hu2 = node2.cost_hessian()
#
# print(c_hx2 - c_hx)







