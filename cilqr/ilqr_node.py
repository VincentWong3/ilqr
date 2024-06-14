from abc import ABC, abstractmethod
import numpy as np


class ILQRNode(ABC):
    def __init__(self, state_dim, control_dim, constraint_dim, goal):
        self._state_dim = state_dim
        self._control_dim = control_dim
        self._constraint_dim = constraint_dim
        self._state = np.zeros(state_dim)
        self._control = np.zeros(control_dim)
        self._constraints = np.zeros(constraint_dim)
        self._goal = np.zeros(state_dim) if goal is None else np.array(goal)

    @property
    def state_dim(self):
        return self._state_dim

    @property
    def control_dim(self):
        return self._control_dim

    @property
    def constraint_dim(self):
        return self._constraint_dim

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if len(value) != self._state_dim:
            raise ValueError("Invalid state dimension")
        self._state = np.array(value)

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, value):
        if len(value) != self._control_dim:
            raise ValueError("Invalid control dimension")
        self._control = np.array(value)

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, value):
        if len(value) != self._constraint_dim:
            raise ValueError("Invalid constraint dimension")
        self._constraints = np.array(value)

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        if len(value) != self._state_dim:
            raise ValueError("Invalid goal dimension")
        self._goal = np.array(value)

    @abstractmethod
    def dynamics(self, state, control):
        """Compute the next state based on the current state and control."""
        pass

    @abstractmethod
    def cost(self):
        """Compute the cost associated with the current state and control."""
        pass

    @abstractmethod
    def dynamics_jacobian(self, state=None, control=None):
        """Compute the Jacobian of the dynamics function."""
        pass

    @abstractmethod
    def cost_jacobian(self):
        """Compute the Jacobian of the cost function."""
        pass

    @abstractmethod
    def cost_hessian(self):
        """Compute the Hessian of the cost function."""
        pass

    @abstractmethod
    def constraint_jacobian(self):
        """Compute the Jacobian of the constraint function."""
        pass
