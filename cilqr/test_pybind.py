import sys
import numpy as np
sys.path.append("/home/vincent/ilqr/cilqr/al_ilqr_cpp/bazel-bin")
import ilqr_pybind
import copy
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_s_shape_goal_full(v, dt, num_points):
    goals = []
    for i in range(num_points + 1):
        t = i * dt
        x = v * t
        y = 50 * np.sin(0.1 * t)
        theta = np.arctan2(50 * 0.1 * np.cos(0.1 * t), v)
        dx = v
        dy = 50 * 0.1 * np.cos(0.1 * t)
        ddx = 0
        ddy = -50 * 0.1 * 0.1 * np.sin(0.1 * t)
        curvature = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
        delta = np.arctan(curvature * 1.0)
        goal_state = np.array([x, y, theta, delta, v, 0])  # (x, y, theta, delta, v_desire, a_desire)
        goals.append(goal_state)
    return goals

def generate_cycle_equations(centre_x, centre_y, r, x_dims):
    # 创建矩阵 Q, A, C
    Q = np.zeros((x_dims, x_dims))
    A = np.zeros((1, x_dims))
    C = np.zeros((1, 1))

    # 设置矩阵 C 的值
    C[0, 0] = r * r - centre_x * centre_x - centre_y * centre_y

    # 设置矩阵 Q 和 A 的值
    Q[0, 0] = -1.0
    Q[1, 1] = -1.0
    A[0, 0] = 2 * centre_x
    A[0, 1] = 2 * centre_y

    return Q, A, C

# 使用generate_s_shape_goal_full生成目标状态
v = 10
dt = 0.1
L = 3
k = 0.001
num_points = 30
goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

goal_x = [goal[0] for goal in goal_list_full]
goal_y = [goal[1] for goal in goal_list_full]

# 定义参数
state_dim = 6
control_dim = 2
horizon = 30



Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6]) * 1e3
R = np.array([[1, 0], [0, 1]]) * 1e2

state_min = np.array([-1000, -1000, -2 * np.pi, -10, -100, -10])
state_max = np.array([1000, 1000, 2 * np.pi, 10, 100, 10])
control_min = np.array([-0.2, -1])
control_max = np.array([0.2, 1])

constraints = ilqr_pybind.BoxConstraints6_2(state_min, state_max, control_min, control_max)


ns = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(L, dt, k, goal_list_full[0], Q, R, constraints)

# 创建约束对象

# 创建 ilqr_nodes 列表
ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewBicycleNodeBoxConstraints6_2(L, dt, k, goal_list_full[i], Q, R, constraints)
    ilqr_nodes_list.append(node)


# 初始化状态
init_state = np.array([0, 0, 0, 0, v, 0])


# 创建 NewALILQR 实例
al_ilqr = ilqr_pybind.NewALILQR6_2(ilqr_nodes_list, init_state)

# 设置优化参数
max_outer_iter = 50
max_inner_iter = 100
max_violation = 1e-4

# 调用 optimize 函数
al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

# 获取优化后的状态和控制序列
x_list = al_ilqr.get_x_list()
u_list = al_ilqr.get_u_list()

plot_x = x_list[0,:]
plot_y = x_list[1,:]



Q_list = []

for i in range(5):
    Q_signal = np.zeros((6, 6))
    Q_list.append(Q_signal)

A = np.zeros((5,6))
B = np.array(([0,0],[1,0],[0,1],[-1,0],[0,-1]))
C = np.array(([0],[-0.4],[-1],[-0.4],[-1]))

circle_x = 30
circle_y = 11
circle_r = 6
Qc, Ac, Cc = generate_cycle_equations(circle_x, circle_y, circle_r, 6)
Q_list[0] = Qc
C[0, 0] = Cc.item()
A[0, :] = Ac



quadratic_constraints = ilqr_pybind.QuadraticConstraints6_2_5(Q_list,A,B,C)

quadratic_ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewBicycleNodeQuadraticConstraints6_2_5(L, dt, k, goal_list_full[i], Q, R, quadratic_constraints)
    quadratic_ilqr_nodes_list.append(node)

q_al_ilqr = ilqr_pybind.NewALILQR6_2(quadratic_ilqr_nodes_list, init_state)

q_al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

q_x_list = q_al_ilqr.get_x_list()
q_u_list = q_al_ilqr.get_u_list()

q_plot_x = q_x_list[0,:]
q_plot_y = q_x_list[1,:]

plt.figure(figsize=(10, 6))
ax = plt.gca()

circle = patches.Circle((circle_x, circle_y), circle_r, edgecolor='green', facecolor='lightblue', alpha=0.5)
ax.add_patch(circle)

plt.plot(plot_x, plot_y, label='Optimized State Trajectory (Full)', c='b', marker='o')
plt.plot(goal_x, goal_y, label='init State Trajectory (Full)', c='r', marker='o')
plt.plot(q_plot_x, q_plot_y, label='obs (Full)', c='g', marker='o')


ax.set_aspect('equal')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()




