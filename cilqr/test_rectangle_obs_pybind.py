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



Q = np.diag([1e-1, 1e-1, 1e-0, 1e-9]) * 1e3
R = np.array([[1.0]]) * 1e2

init_state = np.array([0, 0, 0, 0])

# 设置优化参数
max_outer_iter = 50
max_inner_iter = 100
max_violation = 1e-4

A = np.zeros((6, 4))
B = np.zeros((6,1))
C = np.zeros((6, 1))

B[0, 0] = 1
B[1, 0] = -1

C[0, 0] = -0.6
C[1, 0] = -0.6

linear_constraints = ilqr_pybind.LinearConstraints4_1_6(A, B, C)

linear_constraints.set_current_constraints_index(1)

left_obs = []
right_obs = []

obs1 = np.array([[32, 32, 28, 28],
                    [13, 15, 15, 13]])

obs2 = np.array([[18, 18, 14, 14],
                 [3, 7, 7, 3]])

left_obs.append(obs1)
right_obs.append(obs2)
#right_obs.append(obs2)




ilqr_nodes_list = []
for i in range(horizon + 1):
    node = ilqr_pybind.NewLatBicycleNodeLinearConstraints4_1_6(L, dt, k, v, goal_list_full[i], Q, R, linear_constraints)
    ilqr_nodes_list.append(node)

al_ilqr = ilqr_pybind.NewALILQR4_1(ilqr_nodes_list, init_state, left_obs, right_obs)

al_ilqr.optimize(max_outer_iter, max_inner_iter, max_violation)

x_list = al_ilqr.get_x_list()
u_list = al_ilqr.get_u_list()

plot_x = x_list[0,:]
plot_y = x_list[1,:]

plt.figure(figsize=(10, 6))
ax = plt.gca()

rectangle = patches.Rectangle((30, 14), 4, 2,
                              edgecolor='blue', facecolor='lightblue', alpha=0.5)

rectangle2 = patches.Rectangle((16, 5), 4, 4,
                              edgecolor='blue', facecolor='lightblue', alpha=0.5)

ax.add_patch(rectangle)

ax.add_patch(rectangle2)



plt.plot(goal_x, goal_y, label='init State Trajectory (Full)', c='r', marker='o')
plt.plot(plot_x, plot_y, label='obs (Full)', c='g', marker='o')


ax.set_aspect('equal')

plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid()
plt.show()




