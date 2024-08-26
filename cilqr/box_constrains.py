import numpy as np
from linear_constraints import *


class BoxConstraint(LinearConstraints):
    def __init__(self, state_min, state_max, control_min, control_max):
        # 计算约束维度
        state_dim = state_min.size
        control_dim = control_min.size

        # 创建A, B, C矩阵
        # 状态约束部分
        A_state = np.vstack((np.eye(state_dim), -np.eye(state_dim)))
        B_state = np.zeros((2 * state_dim, control_dim))
        C_state = np.hstack((-state_max, state_min))

        # 控制约束部分
        A_control = np.zeros((2 * control_dim, state_dim))
        B_control = np.vstack((np.eye(control_dim), -np.eye(control_dim)))
        C_control = np.hstack((-control_max, control_min))

        # 合并所有约束
        A = np.vstack((A_state, A_control))
        B = np.vstack((B_state, B_control))
        C = np.hstack((C_state, C_control))

        # 确保矩阵形状正确
        assert A.shape == (2 * (state_dim + control_dim), state_dim), f"Shape of A: {A.shape}"
        assert B.shape == (2 * (state_dim + control_dim), control_dim), f"Shape of B: {B.shape}"
        assert C.shape == (2 * (state_dim + control_dim),), f"Shape of C: {C.shape}"

        super().__init__(A, B, C)


# test
#
# state_min = np.array([-1.0, -1.0, -1.0])
# state_max = -state_min
# control_min = np.array([-0.5, -0.5])
# control_max = -control_min
#
# x = np.array([1.5, -1.5, 0.0])
# u = np.array([0.51, -0.51])
#
# cons = BoxConstraint(state_min, state_max, control_min, control_max)
#
# c = cons.constrains(x, u)
#
# a_jx, a_ju = cons.augmented_lagrangian_jacobian(x, u)
# a_hxx, a_huu, a_hxu = cons.augmented_lagrangian_hessian(x, u)
# print(a_jx)







