import sys
import numpy as np
sys.path.append("/home/vincent/ilqr/cilqr/al_ilqr_cpp/bazel-bin")
import ilqr_pybind
import copy
import matplotlib.pyplot as plt

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

# 使用generate_s_shape_goal_full生成目标状态
v = 10
dt = 0.1
L = 3
k = 0.001
num_points = 50
goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

# 定义参数
state_dim = 6
control_dim = 2
horizon = 50



Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6]) * 1e3
R = np.array([[1, 0], [0, 1]]) * 1e2

state_min = np.array([-1000, -1000, -2 * np.pi, -10, -100, -10])
state_max = np.array([1000, 1000, 2 * np.pi, 10, 100, 10])
control_min = np.array([-0.2, -1])
control_max = np.array([0.2, 1])

constraints = ilqr_pybind.BoxConstraints6_2(state_min, state_max, control_min, control_max)


ns = ilqr_pybind.NewBicycleNode6_2(L, dt, k, goal_list_full[0], Q, R, constraints)

# 创建约束对象

# 创建 ilqr_nodes 列表
ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewBicycleNode6_2(L, dt, k, goal_list_full[i], Q, R, constraints)
    ilqr_nodes_list.append(node)

# 将列表转换为元组
ilqr_nodes = tuple(ilqr_nodes_list)

# 初始化状态
init_state = np.array([0, 0, 0, 0, v, 0])


# 创建 NewALILQR 实例
al_ilqr = ilqr_pybind.NewALILQR6_2(ilqr_nodes, init_state)

# 设置优化参数
max_outer_iter = 50
max_inner_iter = 100
max_violation = 1e-4

# 调用 optimize 函数
al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

# 获取优化后的状态和控制序列
x_list = al_ilqr.get_x_list()
u_list = al_ilqr.get_u_list()

