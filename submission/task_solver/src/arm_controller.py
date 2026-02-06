#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手臂控制器模块

该模块实现了手臂控制相关的功能，包括：
1. 手臂位置比较
2. 手臂动作序列控制
3. 任务二的手臂控制逻辑（抓取和放置物体）
4. 任务三的手臂控制逻辑（搬运箱子）
"""

# 控制器路径
import sys
import os

YOLO_MODEL_PATH = "/TongVerse/biped_challenge/submission/best.pt"
CONTROLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTROLLER_PATH, "devel/lib/python3/dist-packages"))
import numpy as np
from typing import Dict, Any, Optional
from submission.task_solver.src.utils import Utils

class ArmController:
    """手臂控制器类，处理手臂相关的控制逻辑"""
    
    def __init__(self, shared_resource):
        """初始化手臂控制器
        
        Args:
            shared_resource: 共享资源实例，用于访问和修改共享变量
        """
        self.shared_resource = shared_resource
        self.utils = Utils(shared_resource)

    def compare_arm_positions(self, current_pos, target_pos, tolerance=0.1):
        """比较手臂位置是否到达目标位置
        
        Args:
            current_pos: 当前手臂位置
            target_pos: 目标手臂位置
            tolerance: 容差
        
        Returns:
            bool: 是否到达目标位置
        """
        return np.allclose(current_pos, target_pos, atol=tolerance)

    def compare_arm_positions_strict(self, current_pos, target_pos):
        """严格比较手臂位置是否到达目标位置（更小的容差）
        
        Args:
            current_pos: 当前手臂位置
            target_pos: 目标手臂位置
        
        Returns:
            bool: 是否到达目标位置
        """
        for i in range(len(current_pos)):
            if abs(current_pos[i] - target_pos[i]) > 0.04:  # 更严格的容差
                return False
        return True

    def set_arm_position(self, action, arm_values):
        """设置手臂位置
        
        Args:
            action: 动作字典
            arm_values: 手臂关节值
        """
        action["arms"]["joint_values"] = arm_values

    def update_arm_sequence(self, action, current_pose_state, arm_values, max_state, on_complete=None):
        """更新手臂动作序列
        
        Args:
            action: 动作字典
            current_pose_state: 当前状态
            arm_values: 手臂动作序列
            max_state: 最大状态值
            on_complete: 完成时的回调函数
        
        Returns:
            int: 更新后的状态值
        """
        if current_pose_state < max_state:
            if self.compare_arm_positions(
                self.shared_resource.current_obs["Kuavo"]["joint_state"]["arms"]["positions"], 
                arm_values[current_pose_state]
            ):
                return current_pose_state + 1
        elif current_pose_state == max_state and on_complete:
            if self.compare_arm_positions(
                self.shared_resource.current_obs["Kuavo"]["joint_state"]["arms"]["positions"], 
                arm_values[current_pose_state]
            ):
                on_complete()
        return current_pose_state

    def handle_tasktwo_arms(self, action):
        """处理任务二的手臂控制逻辑
        
        Args:
            action: 动作字典
        """
        if self.shared_resource.grab == 1:
            if self.shared_resource.tasktwo_pose_state == 3:
                self.shared_resource.tasktwo_pose_state = 2
            
            self.set_arm_position(action, self.shared_resource.tasktwo_arm_values[self.shared_resource.tasktwo_pose_state])
            
            if self.shared_resource.tasktwo_pose_state < 2:
                self.shared_resource.tasktwo_pose_state = self.update_arm_sequence(
                    action, 
                    self.shared_resource.tasktwo_pose_state, 
                    self.shared_resource.tasktwo_arm_values, 
                    2
                )
            elif self.shared_resource.tasktwo_pose_state == 2:
                if self.compare_arm_positions(
                    self.shared_resource.current_obs["Kuavo"]["joint_state"]["arms"]["positions"], 
                    self.shared_resource.tasktwo_arm_values[self.shared_resource.tasktwo_pose_state]
                ):
                    action["pick"] = "left_hand"
                    self.shared_resource.ifpick = True
                    self.shared_resource.is_busy = False
        
        elif self.shared_resource.is_placing:
            action["pick"] = None
            self.shared_resource.ifpick = False
            self.shared_resource.is_busy = False
            self.shared_resource.is_placing = False
        
        elif self.shared_resource.grab == 0 and not self.shared_resource.is_placing:
            self.set_arm_position(action, self.shared_resource.tasktwo_arm_values[self.shared_resource.tasktwo_pose_state])
            if self.shared_resource.tasktwo_pose_state < 3:
                self.shared_resource.tasktwo_pose_state = self.update_arm_sequence(
                    action, 
                    self.shared_resource.tasktwo_pose_state, 
                    self.shared_resource.tasktwo_arm_values, 
                    3
                )

    def handle_taskthree_arms(self, action):
        """处理任务三的手臂控制逻辑
        
        Args:
            action: 动作字典
        """
        if self.shared_resource.taskthree_dest == 9:
            self.set_arm_position(action, self.shared_resource.taskthree_arm_values[self.shared_resource.taskthree_pose_state])
            
            if self.shared_resource.taskthree_pose_state < 5:
                self.shared_resource.taskthree_pose_state = self.update_arm_sequence(
                    action, 
                    self.shared_resource.taskthree_pose_state, 
                    self.shared_resource.taskthree_arm_values, 
                    5
                )
            elif self.shared_resource.taskthree_pose_state == 5:
                if self.compare_arm_positions(
                    self.shared_resource.current_obs["Kuavo"]["joint_state"]["arms"]["positions"], 
                    self.shared_resource.taskthree_arm_values[self.shared_resource.taskthree_pose_state]
                ):
                    self.shared_resource.is_grabbing = True
                    self.shared_resource.is_busy = False
        
        elif (self.shared_resource.taskthree_dest in [16, 8, 17]) and not self.shared_resource.taskthree_arrive:
            self.set_arm_position(action, self.shared_resource.taskthree_arm_values[5])
        
        elif self.shared_resource.taskthree_dest == 17 and self.shared_resource.taskthree_arrive:
            self.set_arm_position(action, self.shared_resource.taskthree_arm_values[self.shared_resource.taskthree_pose_state])
            
            if self.shared_resource.taskthree_pose_state < 9:
                self.shared_resource.taskthree_pose_state = self.update_arm_sequence(
                    action, 
                    self.shared_resource.taskthree_pose_state, 
                    self.shared_resource.taskthree_arm_values, 
                    9
                )
            elif self.shared_resource.taskthree_pose_state == 9:
                if self.compare_arm_positions(
                    self.shared_resource.current_obs["Kuavo"]["joint_state"]["arms"]["positions"], 
                    self.shared_resource.taskthree_arm_values[self.shared_resource.taskthree_pose_state]
                ):
                    self.shared_resource.is_busy = False

    def grab_object(self, obs):
        """抓取物体逻辑
        
        - 解析当前目标类别
        - 处理抓取完成后的状态更新
        - 切换到放置阶段
        """
        # 解析当前要抓的类别
        target_class = self.shared_resource.target_class_name

        if self.shared_resource.ifpick:  # 动作已完成
            self.shared_resource.grabbed_objects.append(target_class)
            self.utils.write_log_to_file(f"抓取成功: {target_class}")
            # 进入放置阶段
            self.shared_resource.stage = "place"
            self.shared_resource.is_grabbing_obj = False
            self.shared_resource.grab = 0
            self.shared_resource.is_examing = False
            self.shared_resource.empty_places.append(self.shared_resource.dest)
            self.shared_resource.dest = (self.shared_resource.dest % 15) + 1
        else:
            # 触发抓取动作序列
            self.shared_resource.is_grabbing_obj = True
            self.shared_resource.is_busy = True
            self.shared_resource.grab = 1

    def place_object(self, obs) -> None:
        """放置物体逻辑
        
        - 处理放置完成后的状态更新
        - 切换到抓取阶段
        - 更新目标位置
        """
        placed_class = self.shared_resource.target_class_name

        if not self.shared_resource.ifpick:  # 手已松开表示放置完成
            self.shared_resource.placed_objects.append(placed_class)
            self.utils.write_log_to_file(f"放置完成: {placed_class}")
            self.shared_resource.stage = "grab"
            self.shared_resource.is_placing = False
            self.shared_resource.is_examing = False
            self.shared_resource.dest = (self.shared_resource.dest % 15) + 1
        else:
            # 正在举手放置过程
            self.shared_resource.is_busy = True
            self.shared_resource.is_placing = True  
