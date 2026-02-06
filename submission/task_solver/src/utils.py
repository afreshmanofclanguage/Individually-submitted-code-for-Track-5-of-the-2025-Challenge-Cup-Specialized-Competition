#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具类模块

该模块提供通用工具函数，包括：
1. 日志记录功能
2. 资源清理功能

设计原则：
- 提供简洁的工具函数，减少代码重复
- 与SharedResource集成，方便访问共享资源
- 提供详细的错误处理和日志记录
"""

# 控制器路径
CONTROLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"

import sys
import os
import datetime
import rospy
import signal

# 控制器路径
YOLO_MODEL_PATH = "/TongVerse/biped_challenge/submission/best.pt"
CONTROLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTROLLER_PATH, "devel/lib/python3/dist-packages"))

class Utils:
    """工具类，提供通用工具函数"""

    def __init__(self, shared_resource=None):
        """初始化工具类
        
        Args:
            shared_resource: 共享资源实例，用于访问全局数据
        """
        # 通用常量
        self.controller_path = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"
        self.yolo_model_path = "/TongVerse/biped_challenge/submission/best.pt"
        
        # 共享资源
        self.shared_resource = shared_resource

    def write_log_to_file(self, information):
        """将信息写入日志文件
        
        Args:
            information: 要记录的信息内容
        """
        # 获取当前时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_content = f"{information} + {current_time}"
        filename = f"running_log{datetime.datetime.now().strftime('%Y%m%d')}.txt"
        
        # 写入日志文件
        with open(filename, "a", encoding="utf-8") as log_file:
            log_file.write(log_content + "\n\n")
            rospy.loginfo(f"日志已写入文件: {filename}")

    def cleanup(self):
        """清理资源
        
        清理启动进程和爬楼梯进程，确保所有子进程都被正确终止
        """
        # 清理启动进程
        if self.shared_resource and self.shared_resource.launch_process is not None:
            try:
                rospy.loginfo("清理启动进程...")
                os.killpg(os.getpgid(self.shared_resource.launch_process.pid), signal.SIGTERM)
                self.shared_resource.launch_process.wait()
                self.shared_resource.launch_process = None
                rospy.loginfo("启动进程已清理")
            except Exception as e:
                rospy.logerr(f"清理启动进程时出错: {str(e)}")
            
        # 清理爬楼梯进程
        if self.shared_resource and self.shared_resource.stair_process is not None:
            try:
                rospy.loginfo("清理爬楼梯进程...")
                os.killpg(os.getpgid(self.shared_resource.stair_process.pid), signal.SIGTERM)
                self.shared_resource.stair_process.wait()
                self.shared_resource.stair_process = None
                rospy.loginfo("爬楼梯进程已清理")
            except Exception as e:
                rospy.logerr(f"清理爬楼梯进程时出错: {str(e)}")