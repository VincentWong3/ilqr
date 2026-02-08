import sympy
from sympy import symbols, cos, sin, tan, tanh, Matrix

# 定义符号
x, y, theta, delta, u = symbols('x y theta delta u')
v, L, k, dt, umax = symbols('v L k dt umax')
state = Matrix([x, y, theta, delta])

# 1. 定义修改后的连续动力学 f(x, u)
# 注意：u1 变成了 umax * tanh(u)
u_real = umax * tanh(u)
f = Matrix([
    v * cos(theta),
    v * sin(theta),
    v * tan(delta) / (L * (1 + k * v**2)),
    u_real
])

# 2. RK2 离散化: x_next = x + dt * f(x + 0.5 * dt * f, u)
mid_state = state + 0.5 * dt * f
# 注意：在 RK2 中，中间步的控制量通常保持不变，或者也做插值，这里按你代码逻辑保持 u
f_mid = f.subs({theta: mid_state[2], delta: mid_state[3]})
next_state = state + dt * f_mid

# 3. 求 Jacobian
Jx = next_state.jacobian(state)
Ju = next_state.jacobian([u])

# 4. 求 Hessian (以 x_next 对 u 的二阶导为例)
Huu = [sympy.diff(next_state[i], u, u) for i in range(4)]

# 打印 C++ 风格的代码（示例：x_next 对 u 的一阶导）
print("// Ju matrix elements:")
for i in range(4):
    print(f"Ju({i}, 0) = {sympy.ccode(Ju[i])};")