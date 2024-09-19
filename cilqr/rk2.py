import sympy as sp

# 定义符号变量
x, y, theta, delta, v, a = sp.symbols('x y theta delta v a')
u1, u2 = sp.symbols('u1 u2')
L, k = sp.symbols('L k')
dt = sp.symbols('dt')

# 定义中间状态符号变量
x_mid, y_mid, theta_mid, delta_mid, v_mid, a_mid = sp.symbols('x_mid y_mid theta_mid delta_mid v_mid a_mid')

# 定义状态向量和控制向量
state = sp.Matrix([x, y, theta, delta, v, a])
control = sp.Matrix([u1, u2])
mid_state = sp.Matrix([x_mid, y_mid, theta_mid, delta_mid, v_mid, a_mid])

# 定义状态方程
f = sp.Matrix([
    v * sp.cos(theta),                 # x_dot
    v * sp.sin(theta),                 # y_dot
    v * sp.tan(delta) / (L * (1 + k * v**2)),  # theta_dot
    u1,                                # delta_dot
    a,                                 # v_dot
    u2                                 # a_dot
])

# 计算连续时间的雅可比矩阵
J_x = f.jacobian(state)
J_u = f.jacobian(control)

# 在中间状态处计算状态方程
f_mid = sp.Matrix([
    v_mid * sp.cos(theta_mid),
    v_mid * sp.sin(theta_mid),
    v_mid * sp.tan(delta_mid) / (L * (1 + k * v_mid**2)),
    u1,
    a_mid,
    u2
])

# 计算中间状态的雅可比矩阵 J_x_mid 和 J_u_mid
J_x_mid = f_mid.jacobian(mid_state)
J_u_mid = f_mid.jacobian(control)

# 符号化推导离散化雅可比矩阵
I = sp.eye(6)  # 单位矩阵

# 离散系统的雅可比矩阵
J_x_discrete = I + dt * J_x_mid * (I + dt * J_x / 2)
J_u_discrete = dt * (J_u_mid + J_x_mid * dt * J_u / 2)

# 输出结果
print("连续时间系统的雅可比矩阵 J_x:")
print(J_x)

print("\n连续时间系统的雅可比矩阵 J_x_mid:")
print(J_x_mid)

print("\n离散化后的雅可比矩阵 J_x:")
sp.pprint(J_x_discrete[0, 2])

print("\n离散化后的雅可比矩阵 J_u:")
sp.pprint(J_u_discrete)
