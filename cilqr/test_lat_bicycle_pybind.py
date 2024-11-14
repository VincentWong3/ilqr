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
        goal_state = np.array([x, y, theta, delta])  # (x, y, theta, delta, v_desire, a_desire)
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
num_points = 50
goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

goal_x = [goal[0] for goal in goal_list_full]
goal_y = [goal[1] for goal in goal_list_full]

# 定义参数
state_dim = 4
control_dim = 1
horizon = 50



Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9]) * 1e6
R = np.array([[1.0]]) * 1e2

init_state = np.array([0, 0, 1, 0])

# 设置优化参数
max_outer_iter = 50
max_inner_iter = 100
max_violation = 1e-4



Q_list = []

for i in range(3):
    Q_signal = np.zeros((4, 4))
    Q_list.append(Q_signal)

A = np.zeros((3,4))
B = np.array(([0],[1],[-1]))
C = np.array(([0],[-0.4],[-0.4]))

circle_x = 30
circle_y = 11
circle_r = 6
Qc, Ac, Cc = generate_cycle_equations(circle_x, circle_y, circle_r, 4)
Q_list[0] = Qc
C[0, 0] = Cc.item()
A[0, :] = Ac



quadratic_constraints = ilqr_pybind.QuadraticConstraints4_1_3(Q_list,A,B,C)

quadratic_ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewLatBicycleNodeQuadraticConstraints4_1_3(L, dt, k, v, goal_list_full[i], Q, R, quadratic_constraints)
    quadratic_ilqr_nodes_list.append(node)

q_al_ilqr = ilqr_pybind.NewALILQR4_1(quadratic_ilqr_nodes_list, init_state)

q_al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

q_x_list = q_al_ilqr.get_x_list()
q_u_list = q_al_ilqr.get_u_list()

q_plot_x = q_x_list[0,:]
q_plot_y = q_x_list[1,:]

plt.figure(figsize=(10, 6))
ax = plt.gca()

circle = patches.Circle((circle_x, circle_y), circle_r, edgecolor='green', facecolor='lightblue', alpha=0.5)
ax.add_patch(circle)

plt.plot(goal_x, goal_y, label='init State Trajectory (Full)', c='r', marker='o')
plt.plot(q_plot_x, q_plot_y, label='obs (Full)', c='g', marker='o')


ax.set_aspect('equal')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()




