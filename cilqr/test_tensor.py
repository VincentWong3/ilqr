#运行这个代码
import sympy as sp
import numpy as np
from fast_bicycle_node import *

# 定义符号变量
x, y, theta, delta, v, a, u1, u2, L, dt, k = sp.symbols('x y theta delta v a u1 u2 L dt k')

# 定义状态和控制向量
state = sp.Matrix([x, y, theta, delta, v, a])
control = sp.Matrix([u1, u2])

# 定义连续时间动力学方程
x_dot = v * sp.cos(theta)
y_dot = v * sp.sin(theta)
theta_dot = v * sp.tan(delta) / (L * (1 + k * v**2))
delta_dot = u1
v_dot = a
a_dot = u2

# 连续时间动力学方程的向量表示
f = sp.Matrix([x_dot, y_dot, theta_dot, delta_dot, v_dot, a_dot])

# 计算 k1
k1 = f

# 计算中间状态
state_mid = state + 0.5 * dt * k1

# 计算中间状态的动力学方程
x_dot_mid = state_mid[4] * sp.cos(state_mid[2])
y_dot_mid = state_mid[4] * sp.sin(state_mid[2])
theta_dot_mid = state_mid[4] * sp.tan(state_mid[3]) / (L * (1 + k * state_mid[4]**2))
delta_dot_mid = u1
v_dot_mid = state_mid[5]
a_dot_mid = u2

# 中间状态动力学方程的向量表示
f_mid = sp.Matrix([x_dot_mid, y_dot_mid, theta_dot_mid, delta_dot_mid, v_dot_mid, a_dot_mid])

# 计算 k2
k2 = f_mid

# 计算离散时间动力学方程
state_next = state + dt * k2

# 计算雅可比矩阵
Jx = state_next.jacobian(state)
Ju = state_next.jacobian(control)

# 输出结果
print("Jacobian with respect to state (Jx):")
sp.pretty_print(Jx[0, 2])

print("\nJacobian with respect to control (Ju):")
sp.pretty_print(Jx[0, 4])

def jacobian_directly(theta, delta, v, a, u1, u2, dt, k, L):
    jx = np.array([[1, 0, -dt * (0.5 * a * dt + v) * np.sin(theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))),
             -0.5 * dt ** 2 * v * (0.5 * a * dt + v) * (np.tan(delta) ** 2 + 1) * np.sin(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))) / (L * (k * v ** 2 + 1)),
             -dt * (0.5 * a * dt + v) * (
                         -1.0 * dt * k * v ** 2 * np.tan(delta) / (L * (k * v ** 2 + 1) ** 2) + 0.5 * dt * np.tan(delta) / (
                             L * (k * v ** 2 + 1))) * np.sin(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))) + dt * np.cos(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))),
             0.5 * dt ** 2 * np.cos(theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1)))],
            [0, 1, dt * (0.5 * a * dt + v) * np.cos(theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))),
             0.5 * dt ** 2 * v * (0.5 * a * dt + v) * (np.tan(delta) ** 2 + 1) * np.cos(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))) / (L * (k * v ** 2 + 1)),
             dt * (0.5 * a * dt + v) * (
                         -1.0 * dt * k * v ** 2 * np.tan(delta) / (L * (k * v ** 2 + 1) ** 2) + 0.5 * dt * np.tan(delta) / (
                             L * (k * v ** 2 + 1))) * np.cos(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))) + dt * np.sin(
                 theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1))),
             0.5 * dt ** 2 * np.sin(theta + 0.5 * dt * v * np.tan(delta) / (L * (k * v ** 2 + 1)))], [0, 0, 1, dt * (
                    0.5 * a * dt + v) * (np.tan(delta + 0.5 * dt * u1) ** 2 + 1) / (L * (k * (0.5 * a * dt + v) ** 2 + 1)),
                                                                                                -dt * k * (
                                                                                                            0.5 * a * dt + v) * (
                                                                                                            1.0 * a * dt + 2 * v) * np.tan(
                                                                                                    delta + 0.5 * dt * u1) / (
                                                                                                            L * (k * (
                                                                                                                0.5 * a * dt + v) ** 2 + 1) ** 2) + dt * np.tan(
                                                                                                    delta + 0.5 * dt * u1) / (
                                                                                                            L * (k * (
                                                                                                                0.5 * a * dt + v) ** 2 + 1)),
                                                                                                -1.0 * dt ** 2 * k * (
                                                                                                            0.5 * a * dt + v) ** 2 * np.tan(
                                                                                                    delta + 0.5 * dt * u1) / (
                                                                                                            L * (k * (
                                                                                                                0.5 * a * dt + v) ** 2 + 1) ** 2) + 0.5 * dt ** 2 * np.tan(
                                                                                                    delta + 0.5 * dt * u1) / (
                                                                                                            L * (k * (
                                                                                                                0.5 * a * dt + v) ** 2 + 1))],
            [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 1]])

    return jx


def jacobianx(dt, L, v, a, delta, k, u1, u2):
    den = k * (0.5 * a * dt + v) * (0.5 * a * dt + v) + 1
    den_square = den * den
    tan_delta = np.tan(delta + 0.5 * dt * u1)
    v_mid = 0.5 * a * dt + v
    d1 = dt * tan_delta / den / L
    d2 = -dt * k * v_mid * v_mid * 2 * tan_delta / den / den / L
    return d1 + d2

def jacobian_matrix(theta, delta, v, a, u1, u2, L, k):
    # 计算矩阵元素
    row1 = [0, 0, -v * np.sin(theta), 0, np.cos(theta), 0]
    row2 = [0, 0, v * np.cos(theta), 0, np.sin(theta), 0]

    tan_delta = np.tan(delta)
    sec_delta_sq = 1 / np.cos(delta)**2  # sec^2(delta) = 1/np.cos^2(delta)
    k_v_sq = k * v**2
    denom = L * (k_v_sq + 1)
    denom_squared = L * (k_v_sq + 1)**2

    row3 = [
        0,
        0,
        0,
        v * sec_delta_sq / denom,
        -2 * k * v**2 * tan_delta / denom_squared + tan_delta / denom,
        0
    ]

    row4 = [0, 0, 0, 0, 0, 0]
    row5 = [0, 0, 0, 0, 0, 1]
    row6 = [0, 0, 0, 0, 0, 0]

    # 组装雅可比矩阵
    J = np.array([row1, row2, row3, row4, row5, row6])

    return J

v_num = 0.506653
a_num = 0.885125
delta_num = 0.116172
u1_num = -0.130367
u2_num = 1.58351
k_num = 0.001
dt_num = 0.1
L_num = 3.0

ans = jacobianx(dt_num, L_num, v_num, a_num, delta_num, k_num, u1_num, u2_num)
#print(ans)

x = np.array([5.77417,1.5339,0.507861,0.118011,10.3187,0.885636])
x_mid = np.array([6.22581, 1.78313, 0.524791, 0.109405, 10.3626, 0.964229])

p = -0.1 * 10.3626 * np.sin(0.524791)
v_mid = 0.5 * x[5] * 0.1 + x[4]
theta_mid = x[2] + 0.5 * 0.1 * x[4] * np.tan(x[3]) / (3.0 * (0.001 * x[4] ** 2 + 1))
delta_mid = x[3] + 0.05 * u1_num
a_mid = x[5] + 0.05 * u2_num
print(f"p{theta_mid} p2{delta_mid} p3{v_mid} p4{a_mid}")

ans2 = jacobian_matrix(x[2], x[3], x[4], x[5], -0.130367, 1.58351, 3.0, 0.001)
ans3 = jacobian_matrix(theta_mid, delta_mid, v_mid, a_mid, -0.130367, 1.58351, 3.0, 0.001)
ans4 = (np.eye(6) + ans3 * 0.1)
ans5 = ans3 @ ans2 * 0.005 + ans3 * 0.1 + np.eye(6)
ans6 = jacobian_directly(x[2], x[3], x[4], x[5], -0.130367, 1.58351, 0.1, 0.001, 3.0)
#print(ans2)
#print(ans3)
#print(ans4)
#print(ans5)
print(ans5)



