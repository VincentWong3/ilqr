import numpy as np
from ilqr_node import ILQRNode

class LatBicycleKinematicNode(ILQRNode):
    def __init__(self, L, dt, v, state_bounds, control_bounds, goal, Q, R):
        state_dim = 4
        control_dim = 1
        constraint_dim = 2 * (state_dim + control_dim)  # State and control constraints
        super().__init__(state_dim, control_dim, constraint_dim, goal)
        self.L = L
        self.dt = dt
        self.v = v
        self.state_max = np.array(state_bounds[1])
        self.state_min = np.array(state_bounds[0])
        self.control_max = np.array(control_bounds[1])
        self.control_min = np.array(control_bounds[0])
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.lambda_ = np.zeros(constraint_dim)  # Lagrange multipliers
        self.mu = 1.0  # Penalty factor
        self.Imu = np.zeros((constraint_dim, constraint_dim))  # Diagonal matrix for penalty factor

    def normalize_angle(self, angle):
        """Normalize angle to be within the interval (-π, π)."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def normalize_state(self):
        """Normalize the angles in the state."""
        self.state[2] = self.normalize_angle(self.state[2])  # Normalize theta
        self.state[3] = self.normalize_angle(self.state[3])  # Normalize delta

    def dynamics(self, state, control):
        x, y, theta, delta = state
        u = control[0]
        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)

        # Runge-Kutta 2nd Order (RK2) method for discretization
        k1 = self.dynamics_continuous(state, u)
        mid_state = state + 0.5 * self.dt * k1
        k2 = self.dynamics_continuous(mid_state, u)

        state_next = state + self.dt * k2
        state_next[2] = self.normalize_angle(state_next[2])  # Normalize theta
        state_next[3] = self.normalize_angle(state_next[3])  # Normalize delta
        return state_next

    def dynamics_continuous(self, state, u):
        x, y, theta, delta = state
        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)
        x_dot = self.v * np.cos(theta)
        y_dot = self.v * np.sin(theta)
        theta_dot = self.v * np.tan(delta) / self.L
        delta_dot = u
        return np.array([x_dot, y_dot, theta_dot, delta_dot])

    def dynamics_jacobian(self, state=None, control=None):
        if state is None:
            state = self.state
        if control is None:
            control = self.control

        x, y, theta, delta = state
        u = control[0]

        theta = self.normalize_angle(theta)
        delta = self.normalize_angle(delta)
        dt = self.dt
        v = self.v
        L = self.L

        # Jacobian with respect to state
        Jx = np.array([
            [1, 0, -dt * v * np.sin(theta + 0.5 * dt * v * np.tan(delta) / L),
             -0.5 * dt ** 2 * v ** 2 * (np.tan(delta) ** 2 + 1) * np.sin(theta + 0.5 * dt * v * np.tan(delta) / L) / L],
            [0, 1, dt * v * np.cos(theta + 0.5 * dt * v * np.tan(delta) / L),
             0.5 * dt ** 2 * v ** 2 * (np.tan(delta) ** 2 + 1) * np.cos(theta + 0.5 * dt * v * np.tan(delta) / L) / L],
            [0, 0, 1, dt * v * (np.tan(delta + 0.5 * dt * u) ** 2 + 1) / L],
            [0, 0, 0, 1]
        ])

        # Jacobian with respect to control
        Ju = np.array([
            [0],
            [0],
            [0.5 * dt ** 2 * v * (np.tan(delta + 0.5 * dt * u) ** 2 + 1) / L],
            [dt]
        ])

        return Jx, Ju

    def cost(self):
        self.normalize_state()
        # Define the quadratic cost function
        state_error = self.state - self.goal
        state_cost = state_error.T @ self.Q @ state_error
        control_cost = self.control.T @ self.R @ self.control
        self.update_Imu()
        constraints = self.constraints()
        constraint_cost = self.lambda_.T @ constraints + 0.5 * self.mu * constraints.T @ self.Imu @ constraints
        return state_cost + control_cost + constraint_cost

    def cost_jacobian(self):
        self.normalize_state()
        # Compute the Jacobian of the cost function
        state_error = self.state - self.goal
        Jx = 2 * self.Q @ state_error
        Ju = 2 * self.R @ self.control
        self.update_Imu()
        constrain_jx, constrain_ju = self.constraint_jacobian()
        Jx = Jx + constrain_jx.T @ (self.lambda_ + self.Imu @ self.constraints())
        Ju = Ju + constrain_ju.T @ (self.lambda_ + self.Imu @ self.constraints())

        return Jx, Ju

    def cost_hessian(self):
        # Compute the Hessian of the cost function
        Hx = 2 * self.Q
        Hu = 2 * self.R
        self.update_Imu()
        constrain_jx, constrain_ju = self.constraint_jacobian()
        Hx = Hx + constrain_jx.T @ self.Imu @ constrain_jx
        Hu = Hu + constrain_ju.T @ self.Imu @ constrain_ju
        return Hx, Hu

    def constraints(self):
        self.normalize_state()
        # Define constraints as g(state, control) <= 0
        state_constraints = np.hstack((self.state - self.state_max, self.state_min - self.state))
        control_constraints = np.hstack((self.control - self.control_max, self.control_min - self.control))

        return np.hstack((state_constraints, control_constraints))

    def constraint_jacobian(self):
        self.normalize_state()
        # Constraint Jacobian for state and control constraints
        state_jacobian = np.vstack((np.eye(self.state_dim), -np.eye(self.state_dim)))
        control_jacobian = np.vstack((np.eye(self.control_dim), -np.eye(self.control_dim)))

        constraint_jacobian = np.zeros((self.constraint_dim, self.state_dim + self.control_dim))
        constraint_jacobian[:2 * self.state_dim, :self.state_dim] = state_jacobian
        constraint_jacobian[2 * self.state_dim:, self.state_dim:] = control_jacobian

        Jx = constraint_jacobian[:, :self.state_dim]
        Ju = constraint_jacobian[:, self.state_dim:]

        return Jx, Ju

    def get_lambda(self):
        return self.lambda_

    def set_lambda(self, new_lambda):
        self.lambda_ = new_lambda

    def get_mu(self):
        return self.mu

    def set_mu(self, new_mu):
        self.mu = new_mu

    def update_lambda(self):
        # Update the Lagrange multipliers using the constraint values
        constraints = self.constraints()
        self.lambda_ = np.maximum(0, self.lambda_ + self.mu * constraints)

    def is_state_within_constraints(self):
        """Check if the current state violates any constraints."""
        self.normalize_state()
        state_violations = (self.state > self.state_max) | (self.state < self.state_min)
        control_violations = (self.control > self.control_max) | (self.control < self.control_min)
        return not np.any(state_violations) and not np.any(control_violations)

    def update_Imu(self):
        constraints = self.constraints()
        for i in range(self.constraint_dim):
            if self.lambda_[i] == 0 and constraints[i] <= 0:
                self.Imu[i, i] = 0
            else:
                self.Imu[i, i] = self.mu

# 示例代码
# state_bounds = np.array([[-10, -10, -np.pi, -1], [10, 10, np.pi, 1]])
# control_bounds = np.array([[-1], [1]])
# goal = np.array([0, 0, 0, 0])
# Q = np.eye(4)
# R = np.eye(1)
#
# node = LatBicycleKinematicNode(L=2.5, dt=0.1, v=5.0, state_bounds=state_bounds, control_bounds=control_bounds,
#                                goal=goal, Q=Q, R=R)
# node.state = np.array([-11.0, 2.0, np.pi / 4, 0.1])
# node.control = np.array([2])
#
# # Example usage of update_Imu
# node.update_lambda()
# print("Imu matrix after updating lambda and mu:")
# print(node.Imu)
