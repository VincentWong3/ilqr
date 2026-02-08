import sympy as sp


x, y, theta, delta, v, u, L, dt, k, umax = sp.symbols('x y theta delta v u L dt k umax')

state = sp.Matrix([x, y, theta, delta])
control = sp.Matrix([u])

x_dot = v * sp.cos(theta)
y_dot = v * sp.sin(theta)
theta_dot = v * sp.tan(delta) / (L * (1 + k * v**2)) # 修正 3.0 为 L
delta_dot = umax * sp.tanh(u)

f = sp.Matrix([x_dot, y_dot, theta_dot, delta_dot])

k1 = f

state_mid = state + 0.5 * dt * k1

x_dot_mid = v * sp.cos(state_mid[2])
y_dot_mid = v * sp.sin(state_mid[2])
theta_dot_mid = v * sp.tan(state_mid[3]) / (L * (1 + k * v**2))
delta_dot_mid = umax * sp.tanh(u) # 控制量 u 在中间步通常保持不变

f_mid = sp.Matrix([x_dot_mid, y_dot_mid, theta_dot_mid, delta_dot_mid])

k2 = f_mid
state_next = state + dt * k2

Jx = state_next.jacobian(state)
Ju = state_next.jacobian(control)

hessian_tensor = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
# Calculate the Hessian tensor
for i in range(4):            # Iterate over each element in state_next
    for j in range(4):        # First state variable for second derivative
        for l in range(4):    # Second state variable for second derivative
            # Compute the second derivative
            hessian_tensor[i][j][l] = sp.diff(state_next[i], state[j], state[l])

# 输出结果示例 (你可以根据需要 print 具体的矩阵)
print("Jacobian Jx:", Jx)
print("Jacobian Ju:", Ju)
print("Hessian (delta component):", hessian_tensor) # 这是最重要的非零项