import numpy as np
from ilqr_node import ILQRNode

class LongitudinalNode(ILQRNode):
    def __init__(self, dt, state_bounds, control_bounds, goal, Q, R):
        state_dim = 3
        control_dim = 1
        constraint_dim = 2 * (state_dim + control_dim)  # State and control constraints
        super().__init__(state_dim, control_dim, constraint_dim, goal)
        self.dt = dt
        self.state_max = np.array(state_bounds[1])
        self.state_min = np.array(state_bounds[0])
        self.control_max = np.array(control_bounds[1])
        self.control_min = np.array(control_bounds[0])
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.lambda_ = np.zeros(constraint_dim)  # Lagrange multipliers
        self.mu = 1.0  # Penalty factor
        self.Imu = np.zeros((constraint_dim, constraint_dim))  # Diagonal matrix for penalty factor

    def dynamics(self, state, control):
        s, v, a = state
        u = control[0]

        # Runge-Kutta 2nd Order (RK2) method for discretization
        k1 = self.dynamics_continuous(state, u)
        mid_state = state + 0.5 * self.dt * k1
        k2 = self.dynamics_continuous(mid_state, u)

        state_next = state + self.dt * k2
        return state_next

    def dynamics_continuous(self, state, u):
        s, v, a = state
        s_dot = v
        v_dot = a
        a_dot = u
        return np.array([s_dot, v_dot, a_dot])

    def dynamics_jacobian(self, state=None, control=None):
        if state is None:
            state = self.state
        if control is None:
            control = self.control

        dt = self.dt

        # Jacobian with respect to state
        Jx = np.array([
            [1, dt, 0.5 * dt**2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        # Jacobian with respect to control
        Ju = np.array([
            [0],
            [0.5 * dt**2],
            [dt]
        ])

        return Jx, Ju

    def cost(self):
        # Define the quadratic cost function
        state_error = self.state - self.goal
        state_cost = state_error.T @ self.Q @ state_error
        control_cost = self.control.T @ self.R @ self.control
        self.update_Imu()
        constraints = self.constraints()
        constraint_cost = self.lambda_.T @ constraints + 0.5 * self.mu * constraints.T @ self.Imu @ constraints
        return state_cost + control_cost + constraint_cost

    def cost_jacobian(self):
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
        # Define constraints as g(state, control) <= 0
        state_constraints = np.hstack((self.state - self.state_max, self.state_min - self.state))
        control_constraints = np.hstack((self.control - self.control_max, self.control_min - self.control))

        return np.hstack((state_constraints, control_constraints))

    def constraint_jacobian(self):
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
# state_bounds = np.array([[-10, -10, -10], [10, 10, 10]])
# control_bounds = np.array([[-1], [1]])
# goal = np.array([0, 0, 0])
# Q = np.eye(3)
# R = np.eye(1)
#
# node = LongitudinalNode(dt=0.1, state_bounds=state_bounds, control_bounds=control_bounds, goal=goal, Q=Q, R=R)
# node.state = np.array([5.0, 0.5, 0.2])
# node.control = np.array([2])
#
# # Example usage of update_Imu
# node.update_lambda()
# print("Imu matrix after updating lambda and mu:")
# print(node.Imu)
