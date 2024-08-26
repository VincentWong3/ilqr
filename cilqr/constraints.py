from abc import ABC, abstractmethod
import numpy as np

class Constraints(ABC):
    def __init__(self, constraint_dim, is_equality=False):
        self.constraint_dim = constraint_dim
        self.c = np.zeros(constraint_dim)
        self.cx = None
        self.cu = None
        self.hx = None
        self.hu = None
        self.hxu = None
        self._lambda = np.zeros(constraint_dim)  # 初始为0的列向量
        self._mu = 1.0  # 假设初值为1
        self.is_equality = is_equality
        self.lambda_proj = np.zeros(constraint_dim)
        self.proj_jac = np.zeros((constraint_dim, constraint_dim))

    @property
    def lambda_(self):
        return self._lambda

    @lambda_.setter
    def lambda_(self, value):
        self._lambda = np.array(value)

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._mu = value

    @abstractmethod
    def constrains(self, x, u):
        pass

    @abstractmethod
    def constrains_jacobian(self, x, u):
        pass

    @abstractmethod
    def constrains_hessian(self, x, u):
        pass

    def projection(self, x):
        return np.minimum(x, 0)

    def projection_jacobian(self, x):
        jac = np.zeros((x.size, x.size))
        for i in range(x.size):
            jac[i, i] = 0 if x[i] > 0 else 1
        return jac

    def projection_hessian(self, x, b):
        return np.zeros((x.size, x.size))

    def augmented_lagrangian_cost(self, x, u):
        self.c = self.constrains(x, u)
        if self.is_equality:
            return 0.5 / self._mu * ((self._lambda - self._mu * self.c).T @ (self._lambda - self._mu * self.c) - self._lambda.T @ self._lambda)
        else:
            self.lambda_proj = self.projection(self._lambda - self._mu * self.c)
            return 0.5 / self._mu * (self.lambda_proj.T @ self.lambda_proj - self._lambda.T @ self._lambda)

    def augmented_lagrangian_jacobian(self, x, u):
        self.c = self.constrains(x, u)
        self.cx, self.cu = self.constrains_jacobian(x, u)
        if self.is_equality:
            factor = self._lambda - self._mu * self.c
            dx = -self.cx.T @ factor
            du = -self.cu.T @ factor
        else:
            self.lambda_proj = self.projection(self._lambda - self._mu * self.c)
            self.proj_jac = self.projection_jacobian(self._lambda - self._mu * self.c)
            dx = -(self.proj_jac @ self.cx).T @ self.lambda_proj
            du = -(self.proj_jac @ self.cu).T @ self.lambda_proj

        return dx.astype(np.float64), du.astype(np.float64)

    def augmented_lagrangian_hessian(self, x, u, full_newton=False):
        self.c = self.constrains(x, u)
        temp_lambda = self._lambda.reshape(-1, 1)
        temp_c = self.c.reshape(-1, 1)

        self.hx, self.hu, self.hxu = self.constrains_hessian(x, u)

        self.cx, self.cu = self.constrains_jacobian(x, u)
        factor = temp_lambda - self._mu * temp_c
        if self.is_equality:
            dxdx = self._mu * ((self.cx.T @ self.cx) - np.einsum('ij,jkl->kl', factor.T, self.hx))
            dxdu = self._mu * ((self.cx.T @ self.cu) - np.einsum('ij,jkl->kl', factor.T, self.hxu))
            dudu = self._mu * ((self.cu.T @ self.cu) - np.einsum('ij,jkl->kl', factor.T, self.hu))
        else:
            self.lambda_proj = self.projection(factor)
            self.proj_jac = self.projection_jacobian(factor)
            jac_proj_cx = self.proj_jac @ self.cx
            jac_proj_cu = self.proj_jac @ self.cu
            dxdx = self._mu * ((jac_proj_cx.T @ jac_proj_cx) - np.einsum('ij,jkl->kl', self.lambda_proj.T, self.hx))
            dxdu = self._mu * ((jac_proj_cx.T @ jac_proj_cu) - np.einsum('ij,jkl->kl', self.lambda_proj.T, self.hxu))
            dudu = self._mu * ((jac_proj_cu.T @ jac_proj_cu) - np.einsum('ij,jkl->kl', self.lambda_proj.T, self.hu))

        return dxdx.astype(np.float64), dudu.astype(np.float64), dxdu.astype(np.float64)

    def update_lambda(self):
        if self.is_equality:
            self._lambda -= self._mu * self.c
        else:
            self._lambda = self.projection(self._lambda - self._mu * self.c)

    def update_mu(self, new_mu):
        self._mu = new_mu

    def max_violation(self, x, u):
        self.c = self.constrains(x, u)
        c_proj = self.projection(self.c)
        dc = self.c - c_proj
        return np.linalg.norm(dc, ord=np.inf)
