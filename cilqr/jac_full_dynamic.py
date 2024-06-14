import sympy as sp

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
sp.pretty_print(Jx)

print("\nJacobian with respect to control (Ju):")
sp.pretty_print(Ju)
