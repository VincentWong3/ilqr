import sympy as sp

# 定义变量
x, y, theta, delta, v, a = sp.symbols('x y theta delta v a')
u1, u2 = sp.symbols('u1 u2')
L, k, dt = sp.symbols('L k dt')


# 定义状态和控制向量
state = sp.Matrix([x, y, theta, delta, v, a])
control = sp.Matrix([u1, u2])

# 定义导数
x_dot = v * sp.cos(theta)
y_dot = v * sp.sin(theta)
theta_dot = v * sp.tan(delta) / (L * (1 + k * v**2))
delta_dot = u1
v_dot = a
a_dot = u2

# 状态导数向量
state_dot = sp.Matrix([x_dot, y_dot, theta_dot, delta_dot, v_dot, a_dot])

# 向前欧拉法离散化
new_state = state + dt * state_dot

# 计算对状态变量的雅可比矩阵
jacobian_state = new_state.jacobian(state)

# 计算对控制变量的雅可比矩阵
jacobian_control = new_state.jacobian(control)

# 合并雅可比矩阵
jacobian_matrix = jacobian_state.row_join(jacobian_control)

# 计算海森张量
hessian_tensors = []
for i in range(len(new_state)):
    hessian_state = []
    for j in range(len(state)):
        hessian_state.append([sp.diff(new_state[i], state[j], state[k]) for k in range(len(state))])
    hessian_tensors.append(hessian_state)

# 输出雅可比矩阵
jacobian_matrix_str = sp.pretty(jacobian_matrix)

# 输出海森张量的第一个元素（仅示例）
hessian_tensor_0_str = sp.pretty(hessian_tensors[2])
#
# print("Jacobian Matrix:")
# print(jacobian_matrix_str)

print("\nHessian Tensor for the first state variable:")
print(hessian_tensor_0_str)
