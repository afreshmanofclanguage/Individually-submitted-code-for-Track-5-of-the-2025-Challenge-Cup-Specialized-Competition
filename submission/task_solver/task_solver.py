#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务求解器模块

该模块是整个系统的核心协调器，负责：
1. 初始化系统组件和子控制器
2. 处理传感器数据和关节命令
3. 严格按照执行任务逻辑（任务一、任务二、任务三）
4. 协调子控制器之间的通信
"""
import sys
import os
# 控制器路径
YOLO_MODEL_PATH = "/TongVerse/biped_challenge/submission/best.pt"
CONTROLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTROLLER_PATH, "devel/lib/python3/dist-packages"))

import threading
import datetime
import importlib
import time
import rospy
import subprocess
from std_msgs.msg import Float32
from kuavo_msgs.msg import jointCmd, sensorsData, armTargetPoses
import numpy as np
from typing import Dict, Any, Optional
import signal
from scipy.spatial.transform import Rotation
from std_srvs.srv import SetBool, SetBoolResponse, Trigger
import termios
import tty
import select
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from kuavo_msgs.srv import SetTagPose, changeArmCtrlMode, changeArmCtrlModeRequest, changeArmCtrlModeResponse
import cv2
import torch

# 导入子控制器和共享资源
from submission.task_solver.src.utils import Utils
from submission.task_solver.src.arm_controller import ArmController
from submission.task_solver.src.motion_controller import MotionController
from submission.task_solver.src.vision_controller import VisionController
from submission.task_solver.src.sharedResource import SharedResource

# YOLOv8模型路径


class TaskSolver:
    """任务求解器类，系统的核心协调器
    
    设计原则：
    - 使用SharedResource单例模式管理所有共享数据，简化模块间通信
    - 采用模块化设计，将不同功能分离到专门的控制器中
    - 遵循ROS编程规范，使用发布/订阅模式进行通信
    - 提供详细的文档和注释，确保代码可维护性
    """

    def __init__(self, task_params, agent_params):
        """初始化控制器
        
        Args:
            task_params: 任务参数，包含任务目标和货架位置等信息
            agent_params: 代理参数，包含代理配置信息
        """
        # 创建SharedResource实例（单例模式）
        self.shared_resource = SharedResource()
        
        # 设置任务参数到共享资源
        self.shared_resource.task_params = task_params
        self.shared_resource.agent_params = agent_params

        # 初始化子控制器，传递SharedResource实例
        self.arm_controller = ArmController(self.shared_resource)
        self.vision_controller = VisionController(self.shared_resource)
        self.motion_controller = MotionController(self.shared_resource)
        self.utils = Utils(self.shared_resource)
        
        # 初始化任务参数
        self._init_task_parameters()

        # 启动ROS相关组件
        self._init_ros_components()

    def _init_task_parameters(self):
        """初始化任务参数
        
        更新任务参数到shared_resource，并确保所有子控制器都使用最新的shared_resource引用
        """
        # 更新任务参数到shared_resource
        self.shared_resource.update_task_parameters()
        
        # 更新子控制器的共享资源引用
        self.arm_controller.shared_resource = self.shared_resource
        self.vision_controller.shared_resource = self.shared_resource
        self.motion_controller.shared_resource = self.shared_resource
        self.utils.shared_resource = self.shared_resource

    def _init_ros_components(self):
        """初始化ROS相关组件
        
        启动仿真，初始化ROS节点，创建发布器、订阅器和服务
        """
        # 启动仿真
        self._start_simulation()
        
        # 初始化ROS节点
        rospy.init_node('demo_controller', anonymous=True)
        
        # 创建发布器
        self.sensor_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=2)
        self.shared_resource.cmd_pose_pub = rospy.Publisher('/cmd_pose', Twist, queue_size=10)  

        # 创建订阅器
        self.joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)
        
        # 设置发布频率
        self.publish_rate = rospy.Rate(self.shared_resource.control_freq)  # 控制频率
        
        # 添加仿真启动服务
        self.sim_start_srv = rospy.Service('sim_start', SetBool, self.sim_start_callback)
        
        # 添加退出处理
        rospy.on_shutdown(self.utils.cleanup)
        
        # 添加频率统计的发布器
        self.freq_pub = rospy.Publisher('/controller_freq', Float32, queue_size=10)

    def _start_simulation(self):
        """启动仿真
        
        启动Isaac Sim仿真环境，设置地面重置选项
        """
        # 是否重置地面
        reset_ground: bool = True
        
        # 构建启动命令
        command = f"bash -c 'source {CONTROLLER_PATH}/devel/setup.bash && roslaunch humanoid_controllers load_kuavo_isaac_sim.launch reset_ground:={str(reset_ground).lower()}'"
        print(f"启动仿真命令: {command}")

        try:
            # 启动进程，使用shell=True允许执行完整的命令字符串
            self.shared_resource.launch_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,  # 直接连接到当前终端
                stderr=None,  # 直接连接到当前终端
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid  # 创建新的进程组，便于后续管理
            )
            rospy.loginfo(f"成功启动命令: {command}")
            
            # 检查进程是否立即失败
            if self.shared_resource.launch_process.poll() is not None:
                raise Exception(f"进程启动失败，返回码: {self.shared_resource.launch_process.returncode}")
                
        except Exception as e:
            rospy.logerr(f"启动命令失败: {str(e)}")
            # 清理失败的进程
            if self.shared_resource.launch_process is not None:
                try:
                    os.killpg(os.getpgid(self.shared_resource.launch_process.pid), signal.SIGTERM)
                except Exception as kill_error:
                    rospy.logerr(f"终止进程失败: {str(kill_error)}")
                finally:
                    self.shared_resource.launch_process = None  
    


    def sim_start_callback(self, req: SetBool) -> SetBoolResponse:
        """仿真启动服务的回调函数
        
        Args:
            req: SetBool请求，data字段为True表示启动仿真，False表示停止仿真
            
        Returns:
            SetBoolResponse: 服务响应，包含操作结果和消息
        """
        # 创建响应对象
        response = SetBoolResponse()
        
        # 更新仿真运行状态
        self.shared_resource.sim_running = req.data
        
        # 记录日志
        if req.data:
            rospy.loginfo("仿真已启动")
        else:
            rospy.loginfo("仿真已停止")
        
        # 设置响应
        response.success = True
        response.message = "仿真控制成功"
        
        return response
    
    def joint_cmd_callback(self, msg: jointCmd) -> None:
        """处理关节命令回调
        
        构建动作字典，处理腿部力矩数据和头部位置数据，
        根据任务状态选择不同的手臂控制方法
        
        Args:
            msg: 关节命令消息，包含关节位置和力矩数据
        """
        # 构建动作字典，按照README.md中的格式
        action = {
            "arms": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(14),  # 14个手臂关节
                "stiffness": [100.0] * 14,     # 关节刚度
                "dampings": [20.2, 20.2, 20.5, 20.5, 10.2, 10.2, 20.1, 20.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1],  # 关节阻尼
            },
            "legs": {
                "ctrl_mode": "effort",
                "joint_values": np.zeros(12),  # 12个腿部关节
                "stiffness": [0.0] * 12,       # 不设置刚度
                "dampings": [0.2] * 12,        # 不设置阻尼
            },
            "head": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(2),   # 2个头部关节
                "stiffness": None,             # 不设置刚度
                "dampings": None,              # 不设置阻尼
            }
        }

        # 处理腿部力矩数据
        for i in range(6):
            action["legs"]["joint_values"][i*2] = msg.tau[i]        # 左腿力矩
            action["legs"]["joint_values"][i*2+1] = msg.tau[i+6]    # 右腿力矩
        
        # 处理头部位置数据（如果有的话）
        if len(msg.joint_q) >= 28:  # 确保消息中包含头部数据
            action["head"]["joint_values"][0] = msg.joint_q[26]  # 头部第一个关节
            action["head"]["joint_values"][1] = msg.joint_q[27]  # 头部第二个关节

        # 处理抓取状态
        if self.shared_resource.current_action and "pick" in self.shared_resource.current_action:
            action["pick"] = self.shared_resource.current_action["pick"]
        elif self.shared_resource.is_grabbing_obj:
            action["pick"] = "left_hand"
        else:
            action["pick"] = None

        # 根据任务状态选择不同的手臂控制方法
        if self.shared_resource.tasktwo_state == 0:
            # 任务二状态：使用任务二的手臂控制
            self.arm_controller.handle_tasktwo_arms(action)
        elif self.shared_resource.tasktwo_state >= 1:
            # 任务三状态：使用任务三的手臂控制
            self.arm_controller.handle_taskthree_arms(action)
        else:
            # 默认手臂控制：直接使用关节命令数据
            # 这里依赖message，所以不能直接由armcontroller控制
            for i in range(7):
                action["arms"]["joint_values"][i*2] = msg.joint_q[i+12]    # 左臂关节位置
                action["arms"]["joint_values"][i*2+1] = msg.joint_q[i+19]  # 右臂关节位置

        # 更新当前动作
        self.shared_resource.current_action = action

    def next_action(self, obs):
        """生成下一个动作
        
        根据当前任务ID执行相应的任务逻辑，
        如果没有生成动作，等待并持续发布传感器数据
        
        Args:
            obs: 观测数据，包含机器人状态和环境信息
        
        Returns:
            dict: 动作字典，包含手臂、腿部和头部的控制命令
        """
        # 检查是否需要清理资源
        if rospy.is_shutdown() or (obs["extras"].get("info") not in ["TaskOne is done", "TaskTwo is done", "TaskThree is done", None, "Time limit reached"]):
            self.utils.write_log_to_file(obs["extras"].get("info"))
            self.utils.cleanup()
        
        # 更新当前观测数据
        self.shared_resource.current_obs = obs
        self.shared_resource.last_obs = obs
        
        # 处理观测数据
        self.process_obs(obs)

        # 根据任务ID执行不同的任务
        if obs["extras"].get("Current_Task_ID") == "TaskOne":
            self.taskone(obs)
        else:
            self.tasktwo(obs)

        # 等待生成动作
        while self.shared_resource.current_action is None and not rospy.is_shutdown():
            # 发布传感器数据
            self.process_obs(self.shared_resource.last_obs, republish=True)
            # 等待一个发布周期
            self.publish_rate.sleep()

        # 发布控制频率
        freq = Float32()
        freq.data = 1
        self.freq_pub.publish(freq) 
        
        # 返回当前动作
        return self.shared_resource.current_action
    
    def taskone(self, obs) -> None:
        """任务一:爬楼梯→穿越沙滩→到达终点
        
        按照以下状态顺序执行：
        0: 前往楼梯 → 1: 爬楼梯 → 2: 前往沙滩初始化点 → 
        3: 前往沙滩起点 → 4: 前往沙滩终点 → 5: 前往最终位置
        
        Args:
            obs: 观测数据，包含机器人当前位置和姿态
        """
        # 获取当前位置和姿态
        current_pos = obs["Kuavo"]["body_state"]["world_position"]
        current_quat = obs["Kuavo"]["body_state"]["world_orient"]
        
        # 记录起始位置
        if self.shared_resource.start_x is None:
            self.shared_resource.start_x = current_pos[0]
        
        # 状态0: 前往楼梯
        if self.shared_resource.taskone_state == 0:
            if self.motion_controller.walk_to_point(
                current_quat, 
                self.shared_resource.TARGET_QUAT, 
                current_pos, 
                self.shared_resource.TASKONE_START_STAIR_POSITION, 
                0.3
            ):
                # 到达楼梯起点，切换到状态1
                self.shared_resource.taskone_state = 1
                
        # 状态1: 爬楼梯
        elif self.shared_resource.taskone_state == 1:
            if self.shared_resource.taskone_API:
                # 调用爬楼梯API
                self.motion_controller.start_stair_climb()
                self.stair_climb_start_time = time.time()
                self.shared_resource.taskone_API = False
                
            # 检查是否完成爬楼梯
            height_reached = abs(current_pos[2] - self.shared_resource.TASKONE_END_STAIR_POSITION[2]) <= self.shared_resource.EPS
            time_elapsed = time.time() - self.stair_climb_start_time
            
            # 当机器人不忙、到达指定高度且时间超过30秒，或时间超过40秒时，认为爬楼梯完成
            if (not self.shared_resource.is_busy and height_reached and time_elapsed > 30) or time_elapsed > 40:
                # 爬楼梯完成，切换到状态2
                self.shared_resource.taskone_state = 2
                
        # 状态2: 前往沙滩初始化点
        elif self.shared_resource.taskone_state == 2:
            if self.motion_controller.walk_to_point(
                current_quat, 
                self.shared_resource.TARGET_QUAT, 
                current_pos,
                self.shared_resource.TASKONE_INITIAL_BEACH_POSITION, 
                0.1
            ):
                # 到达沙滩初始化点，切换到状态3
                self.shared_resource.taskone_state = 3

        # 状态3: 前往沙滩起点
        elif self.shared_resource.taskone_state == 3:
            if self.motion_controller.walk_to_point(
                current_quat, 
                self.shared_resource.TARGET_QUAT, 
                current_pos,
                self.shared_resource.TASKONE_START_BEACH_POSITION, 
                0.1
            ):
                # 到达沙滩起点，切换到状态4
                self.shared_resource.taskone_state = 4

        # 状态4: 前往沙滩终点
        elif self.shared_resource.taskone_state == 4:
            if self.motion_controller.walk_to_point(
                current_quat, 
                self.shared_resource.TARGET_QUAT, 
                current_pos,
                self.shared_resource.TASKONE_END_BEACH_POSITION, 
                0.1
            ):
                # 到达沙滩终点，切换到状态5
                self.shared_resource.taskone_state = 5
                
        # 状态5: 前往最终位置
        elif self.shared_resource.taskone_state == 5:
            self.motion_controller.walk_to_point(
                current_quat, 
                self.shared_resource.TASKONE_END_QUAT, 
                current_pos, 
                self.shared_resource.TASKONE_END_POSITION, 
                0.1, 
                "north", 
                True
            )

    def tasktwo(self, obs) -> None:
        """任务二和任务三的处理逻辑
        
        任务二：检测并抓取指定物品
        任务三：搬运箱子到指定货架
        
        Args:
            obs: 观测数据，包含机器人当前位置和姿态
        """
        # 如果机器人忙，直接返回
        if self.shared_resource.is_busy:
            return 

        # 获取当前位置和姿态
        current_pos = obs["Kuavo"]["body_state"]["world_position"] 
        current_quat = obs["Kuavo"]["body_state"]["world_orient"]

        # 检查是否完成任务二或准备开始任务三
        if self.shared_resource.is_tasktwo_ready or len(self.shared_resource.placed_objects) == 3:
            # 移动到任务三的起始位置
            if self.shared_resource.is_tasktwo_ready or self.motion_controller.walk_to_point(
                current_quat,
                self.motion_controller.get_ideal_direction_values_task2_grabbing(self.shared_resource.dest),
                current_pos,
                self.shared_resource.ITEM_POS[self.shared_resource.dest],
                1.0,
                self.motion_controller.get_ideal_direction_string_task2_grabbing(self.shared_resource.dest)
            ):
                # 标记任务二就绪，切换到任务三
                self.shared_resource.is_tasktwo_ready = True
                self.shared_resource.tasktwo_state = 1   # 切换到任务三
        else:
            # 任务二状态
            self.shared_resource.tasktwo_state = 0

        # 任务二处理逻辑
        if self.shared_resource.tasktwo_state == 0:
            # 获取目标方向和位置
            grabbing_dir = self.motion_controller.get_ideal_direction_values_task2_grabbing(self.shared_resource.dest)
            target_pos = self.shared_resource.ITEM_POS[self.shared_resource.dest]
            grabbing_direction = self.motion_controller.get_ideal_direction_string_task2_grabbing(self.shared_resource.dest)

            # 移动到目标位置
            if self.shared_resource.is_examing or self.motion_controller.walk_to_point(
                current_quat, 
                grabbing_dir, 
                current_pos, 
                target_pos, 
                2, 
                grabbing_direction
            ):
                # 检查是否到达放置位置
                if self.shared_resource.stage == "place" and self.shared_resource.dest == 13:  # 箱子位置
                    self.shared_resource.is_examing = True
                    self.utils.write_log_to_file("到达place位置")
                    # 放置物体
                    self.arm_controller.place_object(obs)

                # 检查是否到达抓取位置
                elif self.shared_resource.stage == "grab" and self.shared_resource.dest not in self.shared_resource.empty_places:
                    # 检查是否正在检查物体或需要检查物体
                    if self.shared_resource.is_examing or self.vision_controller.examine_object(obs):
                        self.shared_resource.is_examing = True

                        # 移动到抓取位置
                        if self.motion_controller.walk_to_point(
                            current_quat, 
                            grabbing_dir,
                            current_pos, 
                            self.shared_resource.GRAB_POS[self.shared_resource.dest], 
                            1, 
                            grabbing_direction
                        ):
                            self.utils.write_log_to_file("到达抓取位置")
                            # 抓取物体
                            self.arm_controller.grab_object(obs) 
                        else:
                            self.utils.write_log_to_file("尝试前往抓取位置")
                    else:
                        # 未找到物体，更新目标位置
                        self.shared_resource.dest = (self.shared_resource.dest % 15) + 1  
                # 更新下一个目标位置
                else:
                    self.shared_resource.dest = (self.shared_resource.dest % 15) + 1
                    
        # 任务三处理逻辑
        elif self.shared_resource.tasktwo_state == 1:
            # 移动到任务三的目标位置
            if self.motion_controller.walk_to_point(
                current_quat,
                self.motion_controller.get_ideal_direction_values_task3(self.shared_resource.taskthree_dest),
                current_pos,
                self.shared_resource.item_pos[self.shared_resource.taskthree_dest],
                1.0,
                self.motion_controller.get_ideal_direction_string_task3(self.shared_resource.taskthree_dest)
            ):
                
                # 更新任务三目标位置
                if self.shared_resource.taskthree_dest == 14:
                    self.shared_resource.taskthree_dest = 13
                elif self.shared_resource.taskthree_dest == 13:
                    self.shared_resource.is_busy = True 
                    self.shared_resource.taskthree_dest = 9 
                elif self.shared_resource.taskthree_dest == 9:
                    self.shared_resource.taskthree_dest = 8
                elif self.shared_resource.taskthree_dest == 8:
                    self.shared_resource.taskthree_dest = 16
                elif self.shared_resource.taskthree_dest == 16:
                    self.shared_resource.taskthree_dest = 17
                elif self.shared_resource.taskthree_dest == 17:
                    # 到达任务三终点
                    self.shared_resource.taskthree_arrive = True
                else:
                    pass

        else:  # 这个状态专门用于试验
            if self.motion_controller.walk_to_point(
                current_quat,
                self.shared_resource.dir_quat[3],
                current_pos,
                [4, current_pos[1], current_pos[2]],
                1.0,
                "west"
            ):
                self.shared_resource.taskthree_dest = 16
                pass

    def process_obs(self, obs: Dict[str, Any], republish=False) -> None:
        """处理观测数据并发布传感器数据
        
        解析观测数据，提取IMU数据和关节数据，
        构建传感器数据消息并发布
        
        Args:
            obs: 观测数据字典，包含IMU和关节状态信息
            republish: 是否为重发布模式，默认为False
        """
        # 创建传感器数据消息
        sensor_data = sensorsData()
        
        # 设置时间戳
        current_time = rospy.Time.now()
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"
        sensor_data.sensor_time = rospy.Duration(self.shared_resource.sensor_time)
        
        # 更新传感器时间
        if not republish:
            self.shared_resource.sensor_time += obs["imu_data"]["imu_time"] - self.shared_resource.last_sensor_time
        self.shared_resource.last_sensor_time = obs["imu_data"]["imu_time"]
        
        # 处理IMU数据
        if "imu_data" in obs:
            imu_data = obs["imu_data"]
            sensor_data.imu_data.acc.x = imu_data["linear_acceleration"][0]
            sensor_data.imu_data.acc.y = imu_data["linear_acceleration"][1]
            sensor_data.imu_data.acc.z = imu_data["linear_acceleration"][2]
            sensor_data.imu_data.gyro.x = imu_data["angular_velocity"][0]
            sensor_data.imu_data.gyro.y = imu_data["angular_velocity"][1]
            sensor_data.imu_data.gyro.z = imu_data["angular_velocity"][2]
            sensor_data.imu_data.quat.w = imu_data["orientation"][0]
            sensor_data.imu_data.quat.x = imu_data["orientation"][1]
            sensor_data.imu_data.quat.y = imu_data["orientation"][2]
            sensor_data.imu_data.quat.z = imu_data["orientation"][3]

        # 处理关节数据
        if "Kuavo" in obs and "joint_state" in obs["Kuavo"]:
            joint_state = obs["Kuavo"]["joint_state"]
            
            # 初始化关节数据数组
            sensor_data.joint_data.joint_q = [0.0] * 28       # 关节位置
            sensor_data.joint_data.joint_v = [0.0] * 28       # 关节速度
            sensor_data.joint_data.joint_vd = [0.0] * 28      # 关节加速度（未使用）
            sensor_data.joint_data.joint_torque = [0.0] * 28  # 关节力矩

            # 处理腿部数据
            if "legs" in joint_state:
                legs_data = joint_state["legs"]
                legs_pos = legs_data["positions"]
                legs_vel = legs_data["velocities"]
                legs_effort = legs_data["applied_effort"]
                
                for i in range(6):
                    # 左腿
                    sensor_data.joint_data.joint_q[i] = legs_pos[i*2]
                    sensor_data.joint_data.joint_v[i] = legs_vel[i*2]
                    sensor_data.joint_data.joint_torque[i] = legs_effort[i*2]
                    # 右腿
                    sensor_data.joint_data.joint_q[i+6] = legs_pos[i*2+1]
                    sensor_data.joint_data.joint_v[i+6] = legs_vel[i*2+1]
                    sensor_data.joint_data.joint_torque[i+6] = legs_effort[i*2+1]

            # 处理手臂数据
            if "arms" in joint_state:
                arms_data = joint_state["arms"]
                arms_pos = arms_data["positions"]
                arms_vel = arms_data["velocities"]
                arms_effort = arms_data["applied_effort"]
                
                for i in range(7):
                    # 左臂
                    sensor_data.joint_data.joint_q[i+12] = arms_pos[i*2]
                    sensor_data.joint_data.joint_v[i+12] = arms_vel[i*2]
                    sensor_data.joint_data.joint_torque[i+12] = arms_effort[i*2]
                    # 右臂
                    sensor_data.joint_data.joint_q[i+19] = arms_pos[i*2+1]
                    sensor_data.joint_data.joint_v[i+19] = arms_vel[i*2+1]
                    sensor_data.joint_data.joint_torque[i+19] = arms_effort[i*2+1]

            # 处理头部数据
            if "head" in joint_state:
                head_data = joint_state["head"]
                head_pos = head_data["positions"]
                head_vel = head_data["velocities"]
                head_effort = head_data["applied_effort"]
                
                for i in range(2):
                    sensor_data.joint_data.joint_q[26+i] = head_pos[i]
                    sensor_data.joint_data.joint_v[26+i] = head_vel[i]
                    sensor_data.joint_data.joint_torque[26+i] = head_effort[i]
        
        # 发布传感器数据
        self.sensor_pub.publish(sensor_data)

    def __del__(self):
        """析构函数，清理资源"""
        self.utils.cleanup()