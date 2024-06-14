import sympy as sp

# 定义符号变量
x, y, theta, delta, u, L, v, dt = sp.symbols('x y theta delta u L v dt')

# 定义状态和控制向量
state = sp.Matrix([x, y, theta, delta])
control = sp.Matrix([u])

# 定义连续时间动力学方程
x_dot = v * sp.cos(theta)
y_dot = v * sp.sin(theta)
theta_dot = v * sp.tan(delta) / L
delta_dot = u

# 连续时间动力学方程的向量表示
f = sp.Matrix([x_dot, y_dot, theta_dot, delta_dot])

# 计算 k1
k1 = f

# 计算中间状态
state_mid = state + 0.5 * dt * k1

# 计算中间状态的动力学方程
x_dot_mid = v * sp.cos(state_mid[2])
y_dot_mid = v * sp.sin(state_mid[2])
theta_dot_mid = v * sp.tan(state_mid[3]) / L
delta_dot_mid = u

# 中间状态动力学方程的向量表示
f_mid = sp.Matrix([x_dot_mid, y_dot_mid, theta_dot_mid, delta_dot_mid])

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
