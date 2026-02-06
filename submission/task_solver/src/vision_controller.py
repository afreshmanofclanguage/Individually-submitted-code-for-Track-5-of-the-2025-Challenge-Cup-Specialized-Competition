#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉控制器模块

该模块实现了视觉相关的功能，包括：
1. YOLOv8模型加载和物体检测
2. 深度信息获取和处理
3. 图像坐标到世界坐标的转换
4. 目标物体识别和验证
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

# YOLOv8模型路径

class VisionController:
    """视觉控制器类，处理与视觉相关的功能"""
    
    def __init__(self, shared_resource):
        """初始化视觉控制器
        
        Args:
            shared_resource: 共享资源实例，用于访问和修改共享变量
        """
        self.shared_resource = shared_resource
        self.yolo_model = None  # YOLO模型实例
        self.load_yolo_model()
        self.utils = Utils(shared_resource)
        
        # 目标物品类别
        self.target_classes = ["scissors", "cans", "plates"]

    def load_yolo_model(self):
        """加载YOLOv8模型，如果未安装则使用TsingHua镜像源安装，并记录日志到文件"""
        # 创建日志目录
        log_dir = "yolo_logs"
        os.makedirs(log_dir, exist_ok=True)
    
        # 创建带时间戳的日志文件名
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"yolo_load_{timestamp}.log")
    
        # 日志记录函数 - 同时输出到控制台和文件
        def log_message(message, level="INFO"):
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_message = f"[{timestamp}] [{level}] {message}"
            print(formatted_message)
        
            # 写入日志文件
            with open(log_filename, "a", encoding="utf-8") as log_file:
                log_file.write(formatted_message + "\n")
    
        try:
            # 首先检查模型文件是否存在
            if not os.path.exists(YOLO_MODEL_PATH):
                log_message(f"错误：模型文件不存在 - {YOLO_MODEL_PATH}", "ERROR")
                return
            
            try:
                from ultralytics import YOLO
                log_message("ultralytics包已安装")
            
                # 尝试加载模型
                try:
                    self.yolo_model = YOLO(YOLO_MODEL_PATH)
                    log_message("YOLO模型加载成功")
                
                    # 添加模型验证
                    try:
                        import numpy as np
                        test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                        results = self.yolo_model(test_img)
                        log_message("模型验证通过: 成功处理测试图像")
                    except Exception as e:
                        log_message(f"模型验证警告: 测试运行时出错 - {str(e)}", "WARNING")
                    
                except Exception as e:
                    log_message(f"模型加载失败: {str(e)}", "ERROR")
            
            except ImportError:
                log_message("未安装ultralytics包，镜像源安装...", "WARNING")
                try:
                    # 使用TsingHua镜像源安装ultralytics
                    subprocess.check_call([
                        sys.executable, 
                        "-m", 
                        "pip", 
                        "install", 
                        "ultralytics==8.2.28", 
                        "-i", 
                        "https://pypi.mirrors.ustc.edu.cn/simple/"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    
                    # 重新导入前确保模块已加载
                    importlib.invalidate_caches()
                    
                    from ultralytics import YOLO
                    log_message("ultralytics包安装成功")
                    
                    # 再次尝试加载模型
                    try:
                        self.yolo_model = YOLO(YOLO_MODEL_PATH)
                        log_message("YOLO模型加载成功")
                        
                        # 添加模型验证
                        try:
                            import numpy as np
                            test_img = np.zeros((640, 640, 3), dtype=np.uint8)
                            results = self.yolo_model(test_img)
                            log_message("模型验证通过: 成功处理测试图像")
                        except Exception as e:
                            log_message(f"模型验证警告: 测试运行时出错 - {str(e)}", "WARNING")
                            
                    except Exception as e:
                        log_message(f"安装后模型加载失败: {str(e)}", "ERROR")
                        
                except subprocess.CalledProcessError:
                    log_message("安装失败：请检查网络连接或镜像源可用性", "ERROR")
                except Exception as e:
                    log_message(f"安装过程中出错: {str(e)}", "ERROR")
        
        except Exception as e:
            log_message(f"加载过程中发生意外错误: {str(e)}", "CRITICAL")
        
        finally:
            log_message(f"模型加载过程完成，日志已保存至: {log_filename}")

    def detect_objects(self, obs) -> list:
        """使用YOLOv8模型检测物品，优先保留未抓取的目标
        
        Args:
            obs: 观测数据字典
        
        Returns:
            list: 检测到的物体列表
        """
        self.shared_resource.detected_objects = []   # 重置为空的列表
        
        # 检查是否有相机数据
        if not obs.get("camera") or obs["camera"]["rgb"] is None:
            rospy.logwarn("未获取到相机RGB数据")
            return self.shared_resource.detected_objects
        
        # 获取RGB图像
        rgb_image = obs["camera"]["rgb"]
        
        # 转换图像格式 (如果需要)
        if rgb_image.shape[-1] == 3:
            # 假设是RGB格式，转换为BGR供OpenCV使用
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = rgb_image
        
        # 使用YOLO模型检测
        if self.yolo_model:
            try:
                # 执行检测
                results = self.yolo_model(bgr_image)
                
                # 置信度阈值
                confidence_threshold = 0.2  # 只保留置信度大于0.2的检测结果
                min_depth_threshold = 0.7  # 只保留深度大于0.7m的检测结果

                # 处理检测结果
                for result in results:
                    # 提取检测框
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    # 应用非极大值抑制 (NMS)
                    indices = cv2.dnn.NMSBoxes(
                        bboxes=boxes.tolist(),
                        scores=confidences.tolist(),
                        score_threshold=confidence_threshold,
                        nms_threshold=0.4  # NMS阈值
                    )
                    
                    # 处理保留的检测结果
                    for i in indices:
                        box = boxes[i]
                        confidence = confidences[i]
                        class_id = class_ids[i]
                        class_name = result.names[class_id]
                        
                        # 只关注目标类别且未抓取的物品
                        if class_name in self.target_classes:
                            # 计算中心点坐标
                            center_x = (box[0] + box[2]) / 2
                            center_y = (box[1] + box[3]) / 2
                            
                            # 获取深度信息
                            depth = self.get_depth_value(obs, center_x, center_y)
                            if depth is None:
                                continue

                            # 直接记录深度值，而不是转换到世界坐标
                            if depth > min_depth_threshold:
                                self.shared_resource.detected_objects.append({
                                    "class_name": class_name,
                                    "depth": depth,  # 相机坐标系下的深度值
                                    "u": center_x,   # 图像坐标u
                                    "v": center_y,   # 图像坐标v
                                    "confidence": confidence
                                })
                                rospy.loginfo(f"检测到: {class_name} @ 深度={depth:.3f}m, 置信度: {confidence:.2f}")
                return self.shared_resource.detected_objects
            except Exception as e:
                rospy.logerr(f"物体检测失败: {str(e)}")
                return self.shared_resource.detected_objects
        else:
            rospy.logwarn("YOLO模型未加载，无法检测物体")
            return self.shared_resource.detected_objects
    
    def examine_object(self, obs) -> bool:
        """检查最近的物体是否为目标类别
        
        判断当前检测中最近物体是否属于目标类别，并记录日志
        
        Args:
            obs: 观测数据字典
        
        Returns:
            bool: 最近的物体是否为目标类别
        """
        logs = ["==== 最近物体判定 ===="]

        detected_objects = self.detect_objects(obs)
        if not detected_objects:
            logs.append("无检测 -> False")
            self.print_in_new_terminal("\n".join(logs), "物体检测")
            return False

        target_class = self.shared_resource.target_class_name  # 仅读
        logs.append(f"目标类别: {target_class if target_class else '未设置(恒 False)'}")
        logs.append(f"检测数量: {len(detected_objects)}")

        for obj in detected_objects:
            logs.append(f"- {obj['class_name']} conf={obj['confidence']:.2f} "
                        f"depth={obj['depth']:.2f}")

        nearest_object = min(detected_objects, key=lambda x: x["depth"])  # 按深度排序找到最近的物体
        logs.append(f"最近物体: {nearest_object['class_name']} depth={nearest_object['depth']:.2f}")

        if not target_class:
            logs.append("无目标类别 -> False")
            self.print_in_new_terminal("\n".join(logs), "物体检测")
            return False

        if nearest_object["class_name"] == target_class:
            logs.append("匹配成功 -> True")
            self.print_in_new_terminal("\n".join(logs), "物体检测")
            return True
        else:
            logs.append("匹配失败 -> False")
            self.print_in_new_terminal("\n".join(logs), "物体检测")
            return False

    def get_depth_value(self, obs, x, y) -> Optional[float]:
        """获取深度图中指定坐标的深度值
        
        Args:
            obs: 观测数据字典
            x: 图像x坐标
            y: 图像y坐标
        
        Returns:
            Optional[float]: 深度值，如果获取失败则返回None
        """
        if not obs.get("camera") or obs["camera"]["depth"] is None:
            rospy.logwarn("未获取到深度图数据")
            return None
        
        depth_map = obs["camera"]["depth"]
        height, width = depth_map.shape[:2]
        
        # 确保坐标在图像范围内
        x_clamped = int(np.clip(x, 0, width-1))
        y_clamped = int(np.clip(y, 0, height-1))
        
        # 获取深度值
        depth_value = depth_map[y_clamped, x_clamped]
        
        # 检查深度值是否有效
        if depth_value <= 0 or depth_value > 10:
            rospy.logwarn(f"无效深度值: {depth_value} @ ({x_clamped}, {y_clamped})")
            return None
        
        return depth_value
    
    def print_in_new_terminal(self, message, title="Object Detection"):
        """在新终端窗口中显示消息
        
        Args:
            message: 要显示的消息
            title: 终端窗口标题
        
        Returns:
            bool: 是否成功创建新终端窗口
        """
        try:
            # 尝试使用xterm
            if self._is_command_available("xterm"):
                command = f"xterm -hold -e 'echo \"{message}\"; sleep 10'"
                subprocess.Popen(command, shell=True)
                return True
            
            # 尝试使用gnome-terminal
            elif self._is_command_available("gnome-terminal"):
                command = f"gnome-terminal -- bash -c 'echo \"{message}\"; sleep 10'"
                subprocess.Popen(command, shell=True)
                return True
            
            # 如果以上都不行，使用tmux创建新窗口
            elif self._is_command_available("tmux"):
                session_name = "detection_session"
                window_name = f"detection_{int(time.time())}"
                command = f"tmux new-session -d -s {session_name} 'echo \"{message}\"; sleep 10'"
                subprocess.Popen(command, shell=True)
                rospy.loginfo(f"使用tmux创建了新窗口: {session_name}")
                return True
            
            # 如果所有方法都失败，使用简单方法
            else:
                rospy.logwarn("无法创建新终端窗口，将在当前终端显示消息")
                print(f"\n{'='*60}\n{title}\n{'='*60}")
                print(message)
                print(f"{'='*60}\n")
                return False
        except Exception as e:
            rospy.logerr(f"创建新终端失败: {str(e)}")
            return False

    def _is_command_available(self, command):
        """检查命令是否可用
        
        Args:
            command: 要检查的命令
        
        Returns:
            bool: 命令是否可用
        """
        try:
            subprocess.check_call(f"which {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
        
    def image_to_world(self, u, v, depth, obs) -> Optional[np.ndarray]:
        """将图像中的一个像素点精确地映射到三维世界中的具体位置
        
        Args:
            u: 图像x坐标
            v: 图像y坐标
            depth: 深度值
            obs: 观测数据字典
        
        Returns:
            Optional[np.ndarray]: 世界坐标下的点，如果转换失败则返回None
        """
        # 相机内参
        fx = self.shared_resource.camera_intrinsics[0, 0]  # x轴焦距
        fy = self.shared_resource.camera_intrinsics[1, 1]  # y轴焦距
        cx = self.shared_resource.camera_intrinsics[0, 2]  # 主点x坐标
        cy = self.shared_resource.camera_intrinsics[1, 2]  # 主点y坐标
        
        # 使用小孔相机模型将2D像素坐标转换为3D相机坐标
        z_cam = depth
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        
        # 相机在机器人坐标系中的位置
        camera_pos_in_robot = np.array([0.0, 0.0, 1])  
        
        # 相机坐标系到机器人坐标系的旋转（假设相机朝前）
        # 旋转矩阵（单位矩阵表示无旋转）
        rotation_cam_to_robot = np.eye(3)
        
        # 相机坐标系下的点转换到机器人坐标系
        point_in_robot = rotation_cam_to_robot @ np.array([x_cam, y_cam, z_cam]) + camera_pos_in_robot
        
        # 获取机器人位姿
        robot_pos = np.array(obs["Kuavo"]["body_state"]["world_position"])
        robot_quat = obs["Kuavo"]["body_state"]["world_orient"]
        
        # 转换为旋转矩阵
        rotation = Rotation.from_quat([
            robot_quat[3],  # w
            robot_quat[0],  # x
            robot_quat[1],  # y
            robot_quat[2]   # z
        ])
        rotation_robot_to_world = rotation.as_matrix()
        
        # 机器人坐标系到世界坐标系
        point_in_world = rotation_robot_to_world @ point_in_robot + robot_pos
        
        return point_in_world