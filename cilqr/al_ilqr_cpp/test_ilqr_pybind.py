import os
import sys

import numpy as np
so_path = os.path.abspath("./bazel-bin")
sys.path.append(so_path)

import ilqr_pybind as ilqr
import math
import matplotlib.pyplot as plt

def generate_s_shape_goal_full(v, dt, num_points):
    """
    生成 S 形参考轨迹
    状态向量: [x, y, theta, delta]
    """
    goals = []
    L = 3.0
    for i in range(num_points + 1):
        t = i * dt
        x = v * t
        y = 50 * math.sin(0.1 * t)
        
        # 计算 theta (航向角)
        dy = 50 * 0.1 * math.cos(0.1 * t)
        dx = v
        theta = math.atan2(dy, dx)
        
        # 计算期望前轮转角 delta (基于曲率)
        ddy = -50 * 0.1 * 0.1 * math.sin(0.1 * t)
        ddx = 0
        curvature = (dx * ddy - dy * ddx) / math.pow(dx * dx + dy * dy, 1.5)
        delta = math.atan(curvature * L)
        
        goals.append(np.array([x, y, theta, delta]))
    return goals

def run_ilqr_test():
    # 1. 参数初始化
    v = 10.0
    dt = 0.1
    L = 3.0
    num_points = 50
    umax = 0.2
    
    # 2. 生成参考轨迹
    goal_list = generate_s_shape_goal_full(v, dt, num_points)
    
    # 3. 权重矩阵设置 (必须是 NumPy 格式)
    Q = np.diag([1e2, 1e2, 1e3, 1e-6]) # 1e-1*1e3, 1e-1*1e3, 1.0*1e3...
    R = np.identity(1) * 1.0
    
    # 4. 构造线性约束 LinearConstraints<4, 1, 10>
    # A*x + B*u + C <= 0
    # 我们按照 C++ 里的设置：设置一个极大的松弛，主要测试 Inner 的软约束性能
    A_mat = np.zeros((10, 4))
    B_mat = np.zeros((10, 1))
    C_mat = np.zeros((10, 1))
    
    B_mat[0, 0] = 1.0
    B_mat[1, 0] = -1.0
    C_mat[0, 0] = -100000.0 
    C_mat[1, 0] = -100000.0
    
    # 调用绑定的 LinearConstraints4_1_10
    constraints = ilqr.LinearConstraints4_1_10(A_mat, B_mat, C_mat)
    constraints.set_current_constraints_index(1)
    
    # 5. 构建 Node 序列
    nodes = []
    for i in range(num_points + 1):
        # 注意：这里调用的字符串必须和 bind_new_lat_bicycle_node_inner 里的字符串完全一致
        node = ilqr.NewLatBicycleNodeLinearConstraints4_1_10(
            L, dt, 0.001, v, umax, goal_list[i], Q, R, constraints
        )
        nodes.append(node)
    
    # 6. 初始化状态 [x, y, theta, delta]
    init_state = np.array([0.0, 0.0, 0.0, 0.0])
    
    # 7. 构造 Solver (NewALILQR4_1)
    # 你的 C++ 构造函数支持 (nodes, init_state, left_obs, right_obs)
    # 这里传空列表作为障碍物
    solver = ilqr.NewALILQR4_1(nodes, init_state, [], [])
    
    # 8. 执行优化 (max_outer, max_inner, max_violation)
    print(">>> 正在启动 iLQR 优化 (9700X 算力加速中)...")
    solver.optimize(50, 100, 1e-3)
    print(">>> 优化完成！")
    
    # 9. 获取结果
    x_res = solver.get_x_list() # 结果通常是 (4, N) 的矩阵
    u_res = solver.get_u_list() # 结果通常是 (1, N-1) 的矩阵
    
    # --- 结果可视化 ---
    goal_array = np.array(goal_list).T
    
    plt.figure(figsize=(12, 8))
    
    # 轨迹对比图
    plt.subplot(2, 1, 1)
    plt.plot(goal_array[0, :], goal_array[1, :], 'r--', label='Reference Path')
    plt.plot(x_res[0, :], x_res[1, :], 'b-', label='iLQR Path')
    plt.title("Path Tracking Performance")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    
    # 控制量图 (物理层面的 u)
    plt.subplot(2, 1, 2)
    # 注意：这里的 u_res 是 Inner 算出的原量，需要手动映射到物理量看看效果
    u_physical = umax * np.tanh(u_res[0, :])
    plt.plot(u_physical, 'g-', label='Control (Mapped Physical u)')
    plt.axhline(y=umax, color='r', linestyle='--', label='Constraint Limit')
    plt.axhline(y=-umax, color='r', linestyle='--')
    plt.title("Control Commands (Steering Rate)")
    plt.ylabel("u [rad/s]")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_ilqr_test()