import numpy as np
import matplotlib.pyplot as plt
from lat_bicycle_node import LatBicycleKinematicNode
from full_bicycle_dynamic_node import FullBicycleDynamicNode
from ilqr import ILQR

# 修改generate_s_shape_goal函数生成6维目标状态
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
        goal_state = [x, y, theta, delta, v, 0]  # (x, y, theta, delta, v_desire, a_desire)
        goals.append(goal_state)
    return goals

# 使用generate_s_shape_goal_full生成目标状态
v = 10
dt = 0.1
L = 1
num_points = 200
goal_list_full = generate_s_shape_goal_full(v, dt, num_points)

# 定义 Q 和 R
Q_full = np.diag([1e-1, 1e-1, 1e-0, 1e-9, 1e-6, 1e-6])
R_full = np.array([[10, 0], [0, 10]])

# 定义状态和控制的范围（设置成较大范围以忽略约束）
state_bounds_full = np.array([[-1000, -1000, -2 * np.pi, -10, -100, -10], [1000, 1000, 2 * np.pi, 10, 100, 10]])
control_bounds_full = np.array([[-0.2, -1], [0.2, 1]])

# 创建 ILQRNode 实例列表
ilqr_nodes_full = [
    FullBicycleDynamicNode(L=L, dt=dt, k=0.0001, state_bounds=state_bounds_full, control_bounds=control_bounds_full,
                           goal=goal, Q=Q_full, R=R_full)
    for goal in goal_list_full
]

# 创建 ILQR 实例
ilqr_full = ILQR(ilqr_nodes_full)

# 设置初始状态
ilqr_nodes_full[0].state = np.array([0, 0, 0, 0, v, 0])

# 优化轨迹
x_init_full, u_init_full, x_opt_full, u_opt_full = ilqr_full.optimize()

# 绘制状态轨迹和目标状态
x_init_traj_full = x_init_full[:, 0]
y_init_traj_full = x_init_full[:, 1]
x_opt_traj_full = x_opt_full[:, 0]
y_opt_traj_full = x_opt_full[:, 1]
goal_x_full = [goal[0] for goal in goal_list_full]
goal_y_full = [goal[1] for goal in goal_list_full]

plt.figure(figsize=(10, 6))
plt.plot(x_init_traj_full, y_init_traj_full, label='Initial State Trajectory (Full)', c='g', marker='o')
plt.plot(x_opt_traj_full, y_opt_traj_full, label='Optimized State Trajectory (Full)', c='b', marker='o')
plt.plot(goal_x_full, goal_y_full, label='Goal Trajectory', c='r', marker='x')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Initial vs Optimized State Trajectory vs Goal Trajectory (Full Model)')
plt.legend()
plt.grid()
plt.show()
