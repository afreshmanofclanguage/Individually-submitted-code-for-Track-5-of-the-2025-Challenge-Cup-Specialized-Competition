#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共享资源类模块

该模块实现了一个单例模式的共享资源类，用于存储所有需要在不同控制器之间共享的变量。
这样可以简化参数传递逻辑，减少代码冗余，提高代码的可维护性。
"""
import sys
import os
# 控制器路径
YOLO_MODEL_PATH = "/TongVerse/biped_challenge/submission/best.pt"
CONTROLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTROLLER_PATH, "devel/lib/python3/dist-packages"))
import numpy as np
from geometry_msgs.msg import Twist

class SharedResource:
    """共享资源类，实现单例模式，存储所有需要共享的变量"""
    # 单例模式实例
    _instance = None
    
    def __new__(cls):
        """单例模式实现，确保全局唯一实例"""
        if cls._instance is None:
            cls._instance = super(SharedResource, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """初始化所有共享变量"""
        # 全局参数
        self.task_params = None  # 任务参数
        self.agent_params = None  # 代理参数
        
        # 常量定义
        self.HEIGHT_TOLERANCE = 0.02  # 高度容差（2cm）
        self.EPS = 0.003  # 浮点数的精准度
        
        # 任务一参数
        self.TASKONE_START_STAIR_POSITION = np.array([7.345084, -0.89484565, 0.49943054])  # 楼梯起点位置
        self.TASKONE_END_STAIR_POSITION = np.array([7.345084, 0.2395187, 0.9676266])  # 楼梯终点位置
        self.TASKONE_INITIAL_BEACH_POSITION = np.array([7.345084, 3.6226907, 0.5136904])  # 沙滩初始位置
        self.TASKONE_START_BEACH_POSITION = np.array([8.020497, 3.6226907, 0.5136904])  # 沙滩起点位置
        self.TASKONE_END_BEACH_POSITION = np.array([8.020497, 7.46675, 0.48288932])  # 沙滩终点位置
        self.TASKONE_END_POSITION = np.array([4.642749, 7.46675, 0.019591428])  # 任务一终点位置
        self.TARGET_QUAT = [0.7148131, -0.01873217, 0.0192524, 0.69879943]  # 目标朝向
        self.TASKONE_END_QUAT = [-0.01666361, 0.0270071, -0.00104613, -0.9994958]  # 任务一终点朝向
        
        # 任务状态
        self.taskone_state = 0  # 任务一的移动状态
        self.taskone_API = True  # True表示爬楼梯API可用
        self.tasktwo_state = 1  # 任务二的状态
        self.tasktwo_dest = 1  # 任务二下一个要去的位置编号
        self.taskthree_dest = 13  # 任务三下一个要去的位置编号
        self.taskthree_arrive = False  # 任务三是否到达目标位置
        
        # 任务二参数
        self.grabbed_objects = []  # 已抓取的物品列表
        self.placed_objects = []  # 已放入箱子的物品列表
        self.current_target = None  # 当前目标物品
        self.target_class_name = None  # 旧字段保留兼容
        self.dest = 1  # 下一个要去的位置编号
        self.is_tasktwo_ready = False  # 任务二是否准备就绪
        self.stage = "grab"  # 当前阶段：grab（抓取）或place（放置）
        self.is_grabbing_obj = False  # 是否正在抓取物体
        self.is_placing = False  # 是否正在放置物体
        self.empty_places = [3, 8, 9, 13, 14, 15]  # 空位置列表
        
        # 位置和方向映射
        self.DIR_QUAT = [
            [],  # 0 未使用
            [0.7148131, -0.01873217, 0.0192524, 0.69879943],  # 1 北
            [0.67789936, 0.01730055, 0.02091876, -0.73465333],  # 2 南
            [-0.0104335186, -0.0268540, -0.0019158523, 0.999584666],  # 3 西
            [0.9992892, 0.00347164, 0.02684573, -0.02623927]  # 4 东
        ]
        
        self.ITEM_POS = [
            [],  # 0 未使用
            [4.1896084, 5.6535482, 0.49933538],  # 1 物品位置1
            [4.1896084, 5.1861000, 0.49931714],  # 2 物品位置2
            [4.1896084, 3.7211926, 0.49933538],  # 3 转弯处，无意义
            [2.8135500, 3.7211926, 0.49951154],  # 4 物品位置4
            [2.1819000, 3.7211926, 0.49951154],  # 5 物品位置5
            [1.4400700, 3.7211926, 0.49951154],  # 6 物品位置6
            [1.0433000, 3.7211926, 0.49951154],  # 7 物品位置7
            [0.1204754, 3.7211926, 0.49951154],  # 8 转弯处，无意义
            [0.1204754, 6.1028944, 0.49971786],  # 9 转弯处，无意义
            [0.9145600, 6.1028944, 0.49971786],  # 10 物品位置10
            [2.4297300, 6.1028944, 0.49971786],  # 11 物品位置11
            [1.6977600, 6.1028944, 0.49971786],  # 12 物品位置12
            [2.2909500, 6.3057050, 0.49979962],  # 13 箱子处
            [2.0215227, 7.0585124, 0.49979962],  # 14 转弯处，无意义
            [4.1896084, 7.0585126, 0.49979962]   # 15 转弯处，无意义
        ]
        
        self.GRAB_POS = [
            [], # 0 未使用
            [3.4695187,5.6535482,0.49933538], # 1 抓取位置1
            [3.4792200,5.1861000,0.49931714], # 2 抓取位置2
            [3.5896084,4.3211926,0.49933538], # 3 转弯处，无意义，没用到
            [2.8135500,4.4404000,0.49951154], # 4 抓取位置4
            [2.1819000,4.4663200,0.49951154], # 5 抓取位置5
            [1.4400700,4.4345100,0.49951154], # 6 抓取位置6
            [1.0433000,4.4095200,0.49951154], # 7 抓取位置7
            [0.1204754,4.3211926,0.49951154], # 8 转弯处，无意义，没用到
            [0.1204754,5.6328944,0.49971786], # 9 转弯处，无意义，没用到
            [0.9145600,5.5119900,0.49971786], # 10 抓取位置10
            [2.4297300,5.4940800,0.49971786], # 11 抓取位置11
            [1.6977600,5.5210500,0.49971786], # 12 抓取位置12
            [2.2909500,6.3057050,0.49979962], # 13 箱子处
            [2.0215227,7.0585124,0.49979962], # 14 转弯处，无意义，没用到
            [3.5896084,7.0585126,0.49979962], # 15 转弯处，无意义，没用到
        ]
        
        self.dir_quat = [
            [], # 0 未使用
            [0.7148131, -0.01873217, 0.0192524, 0.69879943], # 1 北
            [0.67789936,0.01730055,0.02091876,-0.73465333], # 2 南
            [-0.0104335186,-0.0268540,-0.0019158523,0.999584666], # 3 西
            [0.9992892,0.00347164,0.02684573,-0.02623927] # 4 东
        ]

        self.item_pos = [
            [], # 0 未使用
            [3.5896084,5.7050633,0.49933538], # 1 物品位置1
            [3.5896084,5.2371092,0.49931714], # 2 物品位置2
            [3.5896084,4.3211926,0.49933538], # 3 物品位置3
            [2.8638502,4.3211926,0.49951154], # 4 物品位置4
            [2.2347428,4.3211926,0.49951154], # 5 物品位置5
            [1.4913233,4.3211926,0.49951154], # 6 物品位置6
            [1.0946572,4.3211926,0.49951154], # 7 物品位置7

            [0.1204754,4.3211926,0.49951154], # 8 物品位置8
            [0.1204754,5.6328944,0.49971786], # 9 物品位置9

            [0.8636923,5.6328944,0.49971786], # 10 物品位置10
            [1.6466462,5.6328944,0.49971786], # 11 物品位置11
            [2.1546596,5.6328944,0.49971786], # 12 物品位置12
            [2.4215227,6.2589874,0.49979962], # 13 箱子处
            [2.4215227,7.0585124,0.49979962], # 14 物品位置14
            [3.5896084,7.0585126,0.49979962], # 15 物品位置15  
            [], # 16 货架中间位置
            []  # 17 货架位置
        ]
        
        # 箱子位置
        self.box_position = np.array([2.2515113, 6.5817876, 0.49938548])  
        # 到达箱子附近的距离阈值
        self.box_near_distance = 0.4
        
        # 存储当前action
        self.debug_subscriber = True  # 设为 True 开启订阅消息打印
        self.API = True
        self.current_action = None  # 当前动作
        self.current_obs = None  # 当前的观测结果
        self.last_obs = None  # 上一次的观测结果
        self.is_busy = False  # 是否忙碌
        self.current_pos = None  # 当前位置
        self.current_quat = None  # 当前朝向
        self.start_x = None  # 起始x坐标
        
        # 控制参数
        self.control_freq = 100  # 控制频率
        self.dt = 1 / self.control_freq  # 控制周期
        
        # 仿真相关变量
        self.sim_running = True  # 仿真是否运行
        self.sensor_time = 0  # 传感器时间
        self.last_sensor_time = 0  # 上一次传感器时间
        self.is_grab_box_demo = True  # 是否为抓箱子演示
        
        # 添加子进程存储变量
        self.launch_process = None  # 启动进程
        
        # ROS相关组件
        self.cmd_pose_msg = Twist()  # 位置命令消息
        self.cmd_pose_msg.linear.x = 0.0
        self.cmd_pose_msg.linear.y = 0.0
        self.cmd_pose_msg.linear.z = 0.0
        self.cmd_pose_msg.angular.z = 0.0
        self.cmd_pose_pub = None  # 位置命令发布器
        
        # 控制器引用
        self.arm_controller = None  # 手臂控制器
        self.vision_controller = None  # 视觉控制器
        self.motion_controller = None  # 运动控制器
        self.utils = None  # 工具类
        
        # 视觉相关变量
        self.camera_intrinsics = np.array([[235.78143, 0, 128], [0, 235.78143, 128], [0, 0, 1]])  # 相机内参
        self.detected_objects = []  # 检测到的物体
        self.near_distance = 1.5  # 近距离阈值
        self.is_examing = False  # 是否正在检查物体
        
        # 目标物品类别
        self.target_classes = ["scissors", "cans", "plates"]  # 目标物品类别列表
        
        # 手臂控制相关参数
        # 任务二手臂姿态值
        self.tasktwo_arm_values = [
            np.deg2rad([0.0] * 14).tolist(),                # 0 初始姿态
            np.deg2rad([0,0,0,0,90,0,-100,0,90,0,0,0,0,0]).tolist(),  # 1 准备抓取姿态
            np.deg2rad([-35,0,-20,0,-45,0,-35,0,-55,0,0,0,0,0]).tolist(),  # 2 抓取姿态
            np.deg2rad([-50,0,0,0,-45,0,-95,0,-75,0,0,0,0,0]).tolist(),  # 3 提起姿态
            np.deg2rad([-40,0,0,0,-45,0,-70,0,-75,0,0,0,0,0]).tolist()   # 4 放置姿态
        ]       # 01 肩关节 正往后 负往前        
                # 67 肘关节 正向后旋转，负向前旋转   
        
        # 任务三手臂姿态值
        self.taskthree_arm_values = [
            np.deg2rad([0.0] * 14).tolist(),                 # 0 初始姿态
            np.deg2rad([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0]).tolist(),  # 1 准备抓取箱子姿态
            np.deg2rad([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0]).tolist(),     # 2 接近箱子姿态
            np.deg2rad([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0]).tolist(),   # 3 抓取箱子姿态
            np.deg2rad([-20,-20,0,0,-5,5,-50,-50,0,0,0,0,0,0]).tolist(),  # 4 提起箱子姿态
            np.deg2rad([-20,-20,0,0,-10,10,-80,-80,0,0,0,0,0,0]).tolist(),# 5 搬运箱子姿态
            np.deg2rad([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0]).tolist(),   # 6 放置箱子姿态
            np.deg2rad([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0]).tolist(),     # 7 松开箱子姿态
            np.deg2rad([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0]).tolist(),  # 8 恢复初始姿态
            np.deg2rad([0.0] * 14).tolist()                               # 9 最终姿态
        ] 
        
        # 手臂控制状态变量
        self.tasktwo_pose_state = 0  # 任务二姿态状态
        self.taskthree_pose_state = 0  # 任务三姿态状态
        self.grab = 0  # 抓取状态
        self.ifpick = False  # 是否抓取
        self.is_grabbing = False  # 是否正在抓取
        
        # 运动控制器相关变量
        self.current_action_name = None  # 当前动作名称
        self.stair_process = None  # 爬楼梯进程
        
        # 其他
        self.idle_arm_values = [0.0] * 14  # 空闲手臂值
        self.current_arms_position = [0.0] * 14  # 当前手臂位置
        
    def update_task_parameters(self):
        """更新任务参数
        
        根据任务参数设置目标物品和货架位置
        """
        if self.task_params:
            # 任务二目标物品设置
            if "plates" in self.task_params["task_goal"]["TaskTwo"]["goal"]:
                self.current_target = "plates"
            elif "cans" in self.task_params["task_goal"]["TaskTwo"]["goal"]:
                self.current_target = "cans"
            elif "scissors" in self.task_params["task_goal"]["TaskTwo"]["goal"]:
                self.current_target = "scissors"
            else:
                self.current_target = None
            
            self.target_class_name = self.current_target  # 旧字段保留兼容
            
            # 任务三货架位置设置
            if "red_shelf" in self.task_params["task_goal"]["TaskThree"]["goal"]:
                self.item_pos[17] = self.task_params["task_goal"]["TaskThree"]["shelves_world_position"]["red_shelf"]
            elif "yellow_shelf" in self.task_params["task_goal"]["TaskThree"]["goal"]:
                self.item_pos[17] = self.task_params["task_goal"]["TaskThree"]["shelves_world_position"]["yellow_shelf"]
            elif "blue_shelf" in self.task_params["task_goal"]["TaskThree"]["goal"]:
                self.item_pos[17] = self.task_params["task_goal"]["TaskThree"]["shelves_world_position"]["blue_shelf"]
            else:
                print("未在任务目标中找到指定颜色的货架")

            
            # 调整货架位置
            self.item_pos[17][1] += 0.5
            # 设置货架中间位置
            self.item_pos[16] = [self.item_pos[17][0], 4.3211926, self.item_pos[17][2]]
