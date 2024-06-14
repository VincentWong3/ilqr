import numpy as np
import matplotlib.pyplot as plt
from lat_bicycle_node import LatBicycleKinematicNode
from full_bicycle_dynamic_node import FullBicycleDynamicNode
from ilqr import ILQR

# 生成 S 型目标状态列表
def generate_s_shape_goal(v, dt, num_points):
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
        goal_state = [x, y, theta, delta]  # (x, y, theta, delta)
        goals.append(goal_state)
    return goals

# 设定参数
v = 10
dt = 0.1
L = 1
num_points = 30

# 生成目标状态列表
goal_list = generate_s_shape_goal(v, dt, num_points)

# 定义 Q 和 R
Q = np.diag([1e-3, 1e-1, 1e1, 1e-9])
R = np.array([[50]])

# 定义状态和控制的范围（设置成较大范围以忽略约束）
state_bounds = np.array([[-1000, -1000, -2 * np.pi, -10], [1000, 1000, 2 * np.pi, 10]])
control_bounds = np.array([[-0.1], [0.1]])

# 创建 ILQRNode 实例列表
ilqr_nodes_lat = [
    LatBicycleKinematicNode(L=L, dt=dt, v=v, state_bounds=state_bounds, control_bounds=control_bounds,
                            goal=goal, Q=Q, R=R)
    for goal in goal_list
]

# 创建 ILQR 实例
ilqr_lat = ILQR(ilqr_nodes_lat)

# 设置初始状态
ilqr_nodes_lat[0].state = np.array([0, 0, 0, 0])

# 优化轨迹
x_init_lat, u_init_lat, x_opt_lat, u_opt_lat = ilqr_lat.optimize()

# 绘制状态轨迹和目标状态
x_init_traj_lat = x_init_lat[:, 0]
y_init_traj_lat = x_init_lat[:, 1]
x_opt_traj_lat = x_opt_lat[:, 0]
y_opt_traj_lat = x_opt_lat[:, 1]
goal_x_lat = [goal[0] for goal in goal_list]
goal_y_lat = [goal[1] for goal in goal_list]

plt.figure(figsize=(10, 6))
plt.plot(x_init_traj_lat, y_init_traj_lat, label='Initial State Trajectory (Lat)', c='g', marker='o')
plt.plot(x_opt_traj_lat, y_opt_traj_lat, label='Optimized State Trajectory (Lat)', c='b', marker='o')
plt.plot(goal_x_lat, goal_y_lat, label='Goal Trajectory', c='r', marker='x')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Initial vs Optimized State Trajectory vs Goal Trajectory (Lateral Model)')
plt.legend()
plt.grid()
plt.show()

