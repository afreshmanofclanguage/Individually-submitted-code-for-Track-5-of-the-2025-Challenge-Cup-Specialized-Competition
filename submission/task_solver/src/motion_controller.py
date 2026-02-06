#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运动控制器模块

该模块实现了运动控制相关的功能，包括：
1. 通用的走路方法，支持不同方向的控制
2. 爬楼梯规划器的启动和监控
3. 不同任务的理想方向计算
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
import os
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
from submission.task_solver.src.utils import Utils

class MotionController:
    """运动控制器类，处理与运动相关的功能"""
    
    def __init__(self, shared_resource):
        """初始化运动控制器
        
        Args:
            shared_resource: 共享资源实例，用于访问和修改共享变量
        """
        self.shared_resource = shared_resource
        self.utils = Utils(shared_resource)

    def walk_to_point(self, current_quat, target_quat, current_pos, target_pos, speed, direction="north", flag=False) -> bool:
        """通用的走路方法，支持不同方向的控制
        
        Args:
            current_quat: 当前朝向四元数
            target_quat: 目标朝向四元数
            current_pos: 当前位置
            target_pos: 目标位置
            speed: 移动速度
            direction: 方向（north, south, east, west）
            flag: 是否交换x和y方向
        
        Returns:
            bool: 是否到达目标位置
        """
        if self.shared_resource.is_busy:
            rospy.logwarn("不能在忙碌状态下行走")
            return False

        # 计算朝向差
        rotation_current = Rotation.from_quat([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
        rotation_target = Rotation.from_quat([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        current_yaw = rotation_current.as_euler('xyz')[2]
        target_yaw = rotation_target.as_euler('xyz')[2]

        yaw_diff = target_yaw - current_yaw
        if yaw_diff > np.pi: 
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi: 
            yaw_diff += 2 * np.pi
        
        # 计算距离
        current_pos_array = np.array(current_pos)
        distance_to_target = np.linalg.norm(current_pos_array[:2] - target_pos[:2])
        
        if abs(yaw_diff) > 0.05:
            # 需要转向
            self.shared_resource.cmd_pose_msg.linear.x = 0.0
            self.shared_resource.cmd_pose_msg.linear.y = 0.0
            if yaw_diff > 0:
                self.shared_resource.cmd_pose_msg.angular.z = -0.2
            else:
                self.shared_resource.cmd_pose_msg.angular.z = 0.2
        elif distance_to_target > 0.1:
            # 移动
            self.shared_resource.cmd_pose_msg.angular.z = 0.0
            y_diff = target_pos[1] - current_pos[1]
            x_diff = target_pos[0] - current_pos[0]
            
            if abs(x_diff) > 0.05:
                # 沿x轴移动
                if x_diff > 0:
                    # 目标在正x方向，向右移动
                    self.shared_resource.cmd_pose_msg.linear.x = 0.0
                    self.shared_resource.cmd_pose_msg.linear.y = -speed
                else:
                    # 目标在负x方向，向左移动
                    self.shared_resource.cmd_pose_msg.linear.x = 0.0
                    self.shared_resource.cmd_pose_msg.linear.y = speed
            elif abs(y_diff) > 0.05:
                # 沿y轴移动
                if y_diff > 0:
                    # 目标在正y方向，向前移动
                    self.shared_resource.cmd_pose_msg.linear.x = speed
                    self.shared_resource.cmd_pose_msg.linear.y = 0.0
                else:
                    # 目标在负y方向，向后移动
                    self.shared_resource.cmd_pose_msg.linear.x = -speed
                    self.shared_resource.cmd_pose_msg.linear.y = 0.0
            else:
                # 接近目标，减速
                self.shared_resource.cmd_pose_msg.linear.x = 0.1
                self.shared_resource.cmd_pose_msg.linear.y = 0.1
        else:
            # 到达目标位置
            self.shared_resource.cmd_pose_msg.linear.x = 0.0
            self.shared_resource.cmd_pose_msg.linear.y = 0.0
            self.shared_resource.cmd_pose_msg.angular.z = 0.0
            self.shared_resource.cmd_pose_pub.publish(self.shared_resource.cmd_pose_msg)
            return True
        
        # 根据方向调整移动向量
        if direction == "west":
            self.shared_resource.cmd_pose_msg.linear.x, self.shared_resource.cmd_pose_msg.linear.y = \
                self.shared_resource.cmd_pose_msg.linear.y, -self.shared_resource.cmd_pose_msg.linear.x
        elif direction == "south":
            self.shared_resource.cmd_pose_msg.linear.x *= -1
            self.shared_resource.cmd_pose_msg.linear.y *= -1
        elif direction == "east":
            self.shared_resource.cmd_pose_msg.linear.x, self.shared_resource.cmd_pose_msg.linear.y = \
                -self.shared_resource.cmd_pose_msg.linear.y, self.shared_resource.cmd_pose_msg.linear.x
        else:  # north
            pass
        
        if flag:
            self.shared_resource.cmd_pose_msg.linear.x, self.shared_resource.cmd_pose_msg.linear.y = \
                self.shared_resource.cmd_pose_msg.linear.y, self.shared_resource.cmd_pose_msg.linear.x
        
        self.shared_resource.cmd_pose_pub.publish(self.shared_resource.cmd_pose_msg)
        return False
    
    def _monitor_stair_process(self):
        """监控爬楼梯进程状态"""
        # 等待进程结束
        self.shared_resource.stair_process.wait()
        
        # 获取返回码
        return_code = self.shared_resource.stair_process.returncode
        
        # 根据返回码处理结果
        if return_code == 0:
            rospy.loginfo("爬楼梯成功完成")
        else:
            rospy.logerr(f"爬楼梯失败，返回码: {return_code}")
        
        # 重置状态
        self.shared_resource.is_busy = False
        self.shared_resource.current_action_name = None
        self.shared_resource.stair_process = None

    def start_stair_climb(self) -> None:
        """启动爬楼梯规划器"""
        self.shared_resource.is_busy = True
        self.shared_resource.current_action_name = "stair_climbing"
        
        command = f"env -i bash -c 'source {CONTROLLER_PATH}/devel/setup.bash && rosrun humanoid_controllers stairClimbPlanner.py'"
        print(command)
        
        try:
            self.shared_resource.stair_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,
                stderr=None,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            rospy.loginfo("成功启动爬楼梯规划器")
            
        except Exception as e:
            rospy.logerr(f"启动爬楼梯规划器失败: {str(e)}")
            if self.shared_resource.stair_process is not None:
                try:
                    os.killpg(os.getpgid(self.shared_resource.stair_process.pid), signal.SIGTERM)
                except:
                    pass
                self.shared_resource.stair_process = None
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitor_stair_process)
        monitor_thread.daemon = True
        monitor_thread.start()

    def get_ideal_direction_string_task2_grabbing(self, destination):
        """获取任务二抓取阶段的理想方向字符串
        
        Args:
            destination: 目标位置编号
        
        Returns:
            str: 方向字符串 (north, south, east, west)
        """
        if destination in range(1, 4):
            return "west"
        elif destination in range(4, 9):
            return "north"
        elif destination in range(9, 10) or destination in range(13, 15): 
            return "east"
        else:
            return "south"

    def get_ideal_direction_values_task2_grabbing(self, destination):
        """获取任务二抓取阶段的理想方向四元数
        
        Args:
            destination: 目标位置编号
        
        Returns:
            list: 方向四元数
        """
        if destination in range(1, 4):
            return self.shared_resource.DIR_QUAT[3]  # west
        elif destination in range(4, 9):
            return self.shared_resource.DIR_QUAT[1]  # north
        elif destination in range(9, 10) or destination in range(13, 15): 
            return self.shared_resource.DIR_QUAT[4]  # east
        else:
            return self.shared_resource.DIR_QUAT[2]  # south

    def get_ideal_direction_values_task2(self, destination):
        """获取任务二的理想方向四元数
        
        Args:
            destination: 目标位置编号
        
        Returns:
            list: 方向四元数
        """
        if destination in range(1, 4):
            return self.shared_resource.dir_quat[3]  # west
        elif destination in range(4, 9):
            return self.shared_resource.dir_quat[1]  # north
        elif destination in range(9, 10) or destination in range(13, 15): 
            return self.shared_resource.dir_quat[4]  # east
        else:
            return self.shared_resource.dir_quat[2]  # south
    
    def get_ideal_direction_string_task2(self, destination):
        """获取任务二的理想方向字符串
        
        Args:
            destination: 目标位置编号
        
        Returns:
            str: 方向字符串 (north, south, east, west)
        """
        if destination in range(1, 4):
            return "west"
        elif destination in range(4, 9):
            return "north"
        elif destination in range(9, 10) or destination in range(13, 15): 
            return "east"
        else:
            return "south"
        
    def get_ideal_direction_values_task3(self, destination):
        """获取任务三的理想方向四元数
        
        Args:
            destination: 目标位置编号
        
        Returns:
            list: 方向四元数
        """
        if destination == 14 or destination == 16:
            return self.shared_resource.dir_quat[3]  # west
        elif destination == 13 or destination == 9:
            return self.shared_resource.dir_quat[4]  # east
        elif destination == 17 or destination == 8:
            return self.shared_resource.dir_quat[2]  # south
        else:
            return self.shared_resource.dir_quat[1]  # north
    
    def get_ideal_direction_string_task3(self, destination):
        """获取任务三的理想方向字符串
        
        Args:
            destination: 目标位置编号
        
        Returns:
            str: 方向字符串 (north, south, east, west)
        """
        if destination == 14 or destination == 16:
            return "west"
        elif destination == 13 or destination == 9:
            return "east"
        elif destination == 17 or destination == 8:
            return "south"
        else:
            return "north"