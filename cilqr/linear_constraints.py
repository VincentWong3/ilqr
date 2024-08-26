import numpy as np
from constraints import *

class LinearConstraints(Constraints):
    def __init__(self, A, B, C, is_equality=False):
        # 确定约束维度
        constraint_dim = A.shape[0]
        super().__init__(constraint_dim, is_equality)
        self.A = A
        self.B = B
        self.C = C

    def constrains(self, x, u):
        # 实现 A*x + B*u + C
        self.c = self.A @ x + self.B @ u + self.C
        return self.c

    def constrains_jacobian(self, x, u):
        # 对于线性约束，雅可比矩阵是常数矩阵 A 和 B
        self.cx = self.A
        self.cu = self.B
        return self.cx, self.cu

    def constrains_hessian(self, x, u):
        # 对于线性约束，Hessian 矩阵是零矩阵
        self.hx = np.zeros((self.constraint_dim, x.size, x.size))
        self.hu = np.zeros((self.constraint_dim, u.size, u.size))
        self.hxu = np.zeros((self.constraint_dim, x.size, u.size))
        return self.hx, self.hu, self.hxu
