#!/usr/bin/env python3
# -*- coding: utf-8 -*-
CONTORLLER_PATH = "/TongVerse/biped_challenge/demo/kuavo-ros-controldev"

import sys
import os
sys.path.append("/opt/ros/noetic/lib/python3/dist-packages")
sys.path.append("/usr/lib/python3/dist-packages")
sys.path.append(os.path.join(CONTORLLER_PATH, "devel/lib/python3/dist-packages"))
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

# YOLOv8模型路径
YOLO_MODEL_PATH = "/TongVerse/biped_challenge/submission/best.pt"
class TaskSolver:
    # 常量定义
    HEIGHT_TOLERANCE = 0.02  # 高度容差（2cm）
    EPS = 0.003  # 浮点数的精准度
    
    # 任务一参数
    TASKONE_START_STAIR_POSITION = np.array([7.345084, -0.89484565, 0.49943054])
    TASKONE_END_STAIR_POSITION = np.array([7.345084, 0.2395187, 0.9676266])
    TASKONE_INITIAL_BEACH_POSITION = np.array([7.345084, 3.6226907, 0.5136904])
    TASKONE_START_BEACH_POSITION = np.array([8.020497, 3.6226907, 0.5136904])
    TASKONE_END_BEACH_POSITION = np.array([8.020497, 7.46675, 0.48288932])
    TASKONE_END_POSITION = np.array([4.642749, 7.46675, 0.019591428])
    TARGET_QUAT = [0.7148131, -0.01873217, 0.0192524, 0.69879943]
    TASKONE_END_QUAT = [-0.01666361, 0.0270071, -0.00104613, -0.9994958]
    
    # 任务状态
    taskone_state = 0  # the moving state of task one
    taskone_API = True  # True means API is available for stair climbing
    
    # 位置和方向映射
    DIR_QUAT = [
        [],  # 0 Unused
        [0.7148131, -0.01873217, 0.0192524, 0.69879943],  # N
        [0.67789936, 0.01730055, 0.02091876, -0.73465333],  # S
        [-0.0104335186, -0.0268540, -0.0019158523, 0.999584666],  # W
        [0.9992892, 0.00347164, 0.02684573, -0.02623927]  # E
    ]
    
    ITEM_POS = [
        [],  # 0 Unused
        [4.1896084, 5.6535482, 0.49933538],  # 1
        [4.1896084, 5.1861000, 0.49931714],  # 2
        [4.1896084, 3.7211926, 0.49933538],  # 3 转弯处，无意义
        [2.8135500, 3.7211926, 0.49951154],  # 4
        [2.1819000, 3.7211926, 0.49951154],  # 5
        [1.4400700, 3.7211926, 0.49951154],  # 6
        [1.0433000, 3.7211926, 0.49951154],  # 7
        [0.1204754, 3.7211926, 0.49951154],  # 8 转弯处，无意义
        [0.1204754, 6.1028944, 0.49971786],  # 9 转弯处，无意义
        [0.9145600, 6.1028944, 0.49971786],  # 10
        [2.4297300, 6.1028944, 0.49971786],  # 11
        [1.6977600, 6.1028944, 0.49971786],  # 12
        [2.2909500, 6.3057050, 0.49979962],  # 13 箱子处
        [2.0215227, 7.0585124, 0.49979962],  # 14 转弯处，无意义
        [4.1896084, 7.0585126, 0.49979962]   # 15 转弯处，无意义
    ]
    GRAB_POS = [
        [], # 0 Unused
        [3.4695187,5.6535482,0.49933538], # 1
        [3.4792200,5.1861000,0.49931714], # 2
        [3.5896084,4.3211926,0.49933538], # 3 转弯处，无意义，没用到
        [2.8135500,4.4404000,0.49951154], # 4
        [2.1819000,4.4663200,0.49951154], # 5
        [1.4400700,4.4345100,0.49951154], # 6
        [1.0433000,4.4095200,0.49951154], # 7
        [0.1204754,4.3211926,0.49951154], # 8 转弯处，无意义，没用到
        [0.1204754,5.6328944,0.49971786], # 9 转弯处，无意义，没用到
        [0.9145600,5.5119900,0.49971786], # 10
        [2.4297300,5.4940800,0.49971786], # 11
        [1.6977600,5.5210500,0.49971786], # 12
        [2.2909500,6.3057050,0.49979962], # 13 箱子处
        [2.0215227,7.0585124,0.49979962], # 14 转弯处，无意义，没用到
        [3.5896084,7.0585126,0.49979962], # 15 转弯处，无意义，没用到
    ]
    def __init__(self, task_params, agent_params):
        """初始化控制器"""
        # 存储当前action
        self.debug_subscriber = True  # 设为 True 开启订阅消息打印
        self.API = True
        self.current_action = None
        self.current_obs = None # 当前的观测结果
        self.is_busy = False
        self.current_pos = None
        self.current_quat = None
        # 任务参数和代理参数
        self.task_params = task_params
        self.agent_params = agent_params

        # 添加新的成员变量
        self.last_obs = None
        self.last_published_time = None
        self.control_freq = 100 
        self.dt = 1 / self.control_freq
        # 添加子进程存储变量
        self.launch_process = None
       
        # 仿真相关变量
        self.sim_running = True
        self.sensor_time = 0
        self.last_sensor_time = 0
        self.is_grab_box_demo = True
        # 添加按键监听相关变量
        self.old_settings = termios.tcgetattr(sys.stdin)
        
        # === 任务二参数初始化 ===
        # 目标物品类别
        self.target_classes = ["scissors", "cans", "plates"]
        # 已抓取的物品
        self.grabbed_objects = []
        # 已放入箱子的物品
        self.placed_objects = []
        if("plates" in task_params["task_goal"]["TaskTwo"]["goal"]):
            self.current_target = "plates"
        elif("cans" in task_params["task_goal"]["TaskTwo"]["goal"]):
            self.current_target = "cans"
        elif("scissors" in task_params["task_goal"]["TaskTwo"]["goal"]):
            self.current_target = "scissors"
        else :
            self.current_target = None
        self.target_class_name = self.current_target  # 旧字段保留兼容
        self.grab=0
        self.empty_places = [3, 8, 9, 13, 14, 15]
        self.tasktwo_pose_state = 0 # idle
        self.stage = "grab"
        self.ifpick = False
        self.tasktwo_arm_values = [
            np.deg2rad([0.0] * 14).tolist(),                
            np.deg2rad([0,0,0,0,90,0,-100,0,90,0,0,0,0,0]).tolist(),  
            np.deg2rad([-35,0,-20,0,-45,0,-35,0,-55,0,0,0,0,0]).tolist(),  
            
            np.deg2rad([-50,0,0,0,-45,0,-95,0,-75,0,0,0,0,0]).tolist(), 
            np.deg2rad([-40,0,0,0,-45,0,-70,0,-75,0,0,0,0,0]).tolist(),                             
        ]       # 01 肩关节 正往后 负往前       
                # 67 肘关节 正向后旋转，负向前旋转   
        self.is_examing = False  # 是否正在检查物体
        self.near_distance = 1.5
        self.camera_intrinsics = np.array([[235.78143, 0, 128], [0, 235.78143, 128], [0, 0, 1]])
        self.detected_objects = []
        self.is_grabbing_obj = False # 是否正在抓取物体
        self.is_placing = False
        self.dest = 1  # 下一个要去的位置编号
        self.is_tasktwo_ready = False

        # 修改为机器人站在箱子前的位置
        self.box_position = np.array([2.2515113, 6.5817876, 0.49938548])  
        # 到达箱子附近的距离阈值
        self.box_near_distance = 0.4
        # 抓取箱子状态标志
        self.is_grabbing = False
        # YOLO模型
        self.yolo_model = None
        

        reset_ground: bool = True
        command = f"bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && roslaunch humanoid_controllers load_kuavo_isaac_sim.launch reset_ground:={str(reset_ground).lower()}'"
        # what is this
        print(command)

        try:
            # 使用shell=True允许执行完整的命令字符串，并将输出直接连接到当前终端
            self.launch_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,  
                stderr=None,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            rospy.loginfo(f"Successfully started command: {command}")
            
            # 检查进程是否立即失败
            if self.launch_process.poll() is not None:
                raise Exception(f"Process failed to start with return code: {self.launch_process.returncode}")
                
        except Exception as e:
            rospy.logerr(f"Failed to start command: {str(e)}")
            if self.launch_process is not None:
                try:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                except:
                    pass
                self.launch_process = None  


        rospy.init_node('demo_controller', anonymous=True)
        
        # 发布器和订阅器
        self.sensor_pub = rospy.Publisher('/sensors_data_raw', sensorsData, queue_size=2)

        self.cmd_pose_pub = rospy.Publisher('/cmd_pose', Twist, queue_size=10)  
        # 创建Twist消息对象
        self.cmd_pose_msg = Twist()
        self.cmd_pose_msg.linear.x = 0.0
        self.cmd_pose_msg.linear.y = 0.0
        self.cmd_pose_msg.linear.z = 0.0
        self.cmd_pose_msg.angular.z = 0.0
        self.start_x = None  # 新增：记录起始x坐标

        self.joint_cmd_sub = rospy.Subscriber('/joint_cmd', jointCmd, self.joint_cmd_callback)
        
        # 设置发布频率
        self.publish_rate = rospy.Rate(self.control_freq)  # 250的发布频率
        
        # 添加仿真启动服务
        self.sim_start_srv = rospy.Service('sim_start', SetBool, self.sim_start_callback)
        
        # 添加退出处理
        rospy.on_shutdown(self.cleanup)
        
        # 添加频率统计的发布器
        self.freq_pub = rospy.Publisher('/controller_freq', Float32, queue_size=10)


        # 这是用于在times秒内将机器人的手臂调整到values的发布器
        self.arm_pose_pub = rospy.Publisher('kuavo_arm_target_poses', armTargetPoses, queue_size=10)
        self.arm_pose_msg = armTargetPoses()

        self.adjust_times = [3]
        self.idle_arm_values =  [0.0] * 14
        self.lift_values = []
        self.lower_values = []
        self.hand = 0
    
        # 这是用于设置手臂控制模式的发布器
        # 0: 保持姿势
        # 1: 行走时自动摆手
        # 2: 外部控制
        self.arm_traj_change_mode_client = rospy.ServiceProxy("/arm_traj_change_mode", changeArmCtrlMode)
        self.request = changeArmCtrlModeRequest()
        self.request.control_mode = 2
        self.current_arms_position = [0.0] * 14
        

        self.dir_quat = [
            [], # 0 Unused
            [0.7148131, -0.01873217, 0.0192524, 0.69879943], # N
            [0.67789936,0.01730055,0.02091876,-0.73465333], # S
            [-0.0104335186,-0.0268540,-0.0019158523,0.999584666], # W
            [0.9992892,0.00347164,0.02684573,-0.02623927] # E
        ]

        self.item_pos = [
            [], # 0 Unused
            [3.5896084,5.7050633,0.49933538], # 1
            [3.5896084,5.2371092,0.49931714], # 2
            [3.5896084,4.3211926,0.49933538], # 3
            [2.8638502,4.3211926,0.49951154], # 4
            [2.2347428,4.3211926,0.49951154], # 5
            [1.4913233,4.3211926,0.49951154], # 6
            [1.0946572,4.3211926,0.49951154], # 7

            [0.1204754,4.3211926,0.49951154], # 8
            [0.1204754,5.6328944,0.49971786], # 9

            [0.8636923,5.6328944,0.49971786], # 10
            [1.6466462,5.6328944,0.49971786], # 11
            [2.1546596,5.6328944,0.49971786], # 12
            [2.4215227,6.2589874,0.49979962], # 13 box
            [2.4215227,7.0585124,0.49979962], # 14
            [3.5896084,7.0585126,0.49979962], # 15  
            [], #                               16 shelf_position_half
            [], #                               17 shelf_position
        ]
        self.tasktwo_state = 1    # 2 : test value 
        self.tasktwo_dest = 1 # 下一个要去的是几号位置

        self.taskthree_dest = 13 
        self.taskthree_pose_state = 0 # idle
        self.taskthree_arrive = False
        self.taskthree_arm_values = [
            np.deg2rad([0.0] * 14).tolist(),                 # 0
            np.deg2rad([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0]).tolist(),  # 1
            np.deg2rad([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0]).tolist(),     # 2
            np.deg2rad([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0]).tolist(),   # 3
            np.deg2rad([-20,-20,0,0,-5,5,-50,-50,0,0,0,0,0,0]).tolist(),  # 4      
            np.deg2rad([-20,-20,0,0,-10,10,-80,-80,0,0,0,0,0,0]).tolist(),# 5  
           # box carried   
            np.deg2rad([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0]).tolist(),   # 6
            np.deg2rad([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0]).tolist(),     # 7
            np.deg2rad([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0]).tolist(),  # 8
            np.deg2rad([0.0] * 14).tolist()                               # 9
        ]       # 01 肩关节 正往后 负往前       
                # 67 肘关节 正向后旋转，负向前旋转

        # 0 idle   [1,4]   carry the box   [5,8] place the box
 
        if "red_shelf" in task_params["task_goal"]["TaskThree"]["goal"]:
            self.item_pos[17] = task_params["task_goal"]["TaskThree"]["shelves_world_position"]["red_shelf"]
        elif "yellow_shelf" in task_params["task_goal"]["TaskThree"]["goal"]:
            self.item_pos[17] = task_params["task_goal"]["TaskThree"]["shelves_world_position"]["yellow_shelf"]
        elif "blue_shelf" in task_params["task_goal"]["TaskThree"]["goal"]:
            self.item_pos[17] = task_params["task_goal"]["TaskThree"]["shelves_world_position"]["blue_shelf"]
        else: # end is 17
            print("未在任务目标中找到指定颜色的货架")

        self.item_pos[17][1] += 0.5
        self.item_pos[16] = [self.item_pos[17][0],4.3211926,self.item_pos[17][2]] # halfway

        # 加载YOLO模型
        self.load_yolo_model()

        rospy.sleep(2)

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

    def sim_start_callback(self, req: SetBool) -> SetBoolResponse:
        """仿真启动服务的回调函数
        
        Args:
            req: SetBool请求，data字段为True表示启动仿真，False表示停止仿真
            
        Returns:
            SetBoolResponse: 服务响应
        """
        response = SetBoolResponse()
        
        self.sim_running = req.data
        
        if req.data:
            rospy.loginfo("Simulation started")
        else:
            rospy.loginfo("Simulation stopped")
        
        response.success = True
        response.message = "Simulation control successful"
        
        return response
    
    def joint_cmd_callback(self, msg: jointCmd) -> None:
        """处理关节命令回调
        Args:
            msg: 关节命令消息
        """
        # 构建action字典，按照README.md中的格式
        action = {
            "arms": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(14),  # 14 arm joints
                "stiffness": [100.0] * 14 if self.is_grab_box_demo else [200.0] * 14,  # 搬箱子需要更低刚度的手臂
                "dampings": [20.2, 20.2, 20.5, 20.5, 10.2, 10.2, 20.1, 20.1, 10.1, 10.1, 10.1, 10.1, 10.1, 10.1],
            },
            "legs": {
                "ctrl_mode": "effort",
                "joint_values": np.zeros(12),  # 12 leg joints
                "stiffness": [0.0] * 12,  # Not setting stiffness
                "dampings": [0.2] * 12,  # Not setting dampings
            },
            "head": {
                "ctrl_mode": "position",
                "joint_values": np.zeros(2),  # 2 head joints
                "stiffness": None,  # Not setting stiffness
                "dampings": None,  # Not setting dampings
            }
        }

        # 处理腿部力矩数据
        for i in range(6):
            action["legs"]["joint_values"][i*2] = msg.tau[i]        # 左腿
            action["legs"]["joint_values"][i*2+1] = msg.tau[i+6]    # 右腿
        
        # 处理头部位置数据（如果有的话）
        if len(msg.joint_q) >= 28:  # 确保消息中包含头部数据
            action["head"]["joint_values"][0] = msg.joint_q[26]  # 头部第一个关节
            action["head"]["joint_values"][1] = msg.joint_q[27]  # 头部第二个关节

        if self.current_action and "pick" in self.current_action:
            action["pick"] = self.current_action["pick"]
        elif self.is_grabbing_obj == True:
            action["pick"] = "left_hand"
        else:
            action["pick"] = None

        # 更新当前action
        if(self.tasktwo_state == 0 and self.grab == 1):
            # 0 or 3 initially
            if(self.tasktwo_pose_state == 3):
                self.tasktwo_pose_state = 2 

            action["arms"]["joint_values"] = self.tasktwo_arm_values[self.tasktwo_pose_state] 
            if(self.tasktwo_pose_state < 2):
                if(self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.tasktwo_arm_values[self.tasktwo_pose_state])):
                    self.tasktwo_pose_state += 1
            elif(self.tasktwo_pose_state == 2 and self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.tasktwo_arm_values[self.tasktwo_pose_state])):
                action["pick"] = "left_hand"
                self.ifpick = True
                self.is_busy = False
            else:
                pass

        elif(self.tasktwo_state == 0 and self.is_placing == True):
            action["pick"] = None # loose its hand
            self.ifpick = False
            self.is_busy = False
            self.is_placing = False # new 
        elif(self.tasktwo_state == 0 and self.grab == 0 and not self.is_placing): # adjust arm while walking 
            # 2 initially
            action["arms"]["joint_values"] = self.tasktwo_arm_values[self.tasktwo_pose_state] 
            if(self.tasktwo_pose_state < 3): # 3 is the end 
                if(self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.tasktwo_arm_values[self.tasktwo_pose_state])):
                    self.tasktwo_pose_state += 1


        elif(self.tasktwo_state == 1 and self.taskthree_dest == 9):
            # 0 initially
            action["arms"]["joint_values"] = self.taskthree_arm_values[self.taskthree_pose_state] # publish itself

            if(self.taskthree_pose_state < 5): # before carry the box
                if(self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.taskthree_arm_values[self.taskthree_pose_state])):
                    self.taskthree_pose_state += 1
            elif(self.taskthree_pose_state == 5 and self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.taskthree_arm_values[self.taskthree_pose_state])):
                self.is_grabbing = True
                self.is_busy = False
            else:
                pass

        elif(self.tasktwo_state == 1 and (self.taskthree_dest == 16 or self.taskthree_dest == 8 or self.taskthree_dest == 17) and not self.taskthree_arrive):
            # not arriving at the shelf
            action["arms"]["joint_values"] = self.taskthree_arm_values[5]
            pass

        elif(self.tasktwo_state == 1 and self.taskthree_dest == 17 and self.taskthree_arrive): # arrive at the shelf
            # 5 initially
            # self.is_busy = True
            action["arms"]["joint_values"] = self.taskthree_arm_values[self.taskthree_pose_state] # publish itself

            if(self.taskthree_pose_state < 9): # before carry the box
                if(self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.taskthree_arm_values[self.taskthree_pose_state])):
                    self.taskthree_pose_state += 1
            elif(self.taskthree_pose_state == 9 and self.arm_pos_cmp(self.current_obs["Kuavo"]["joint_state"]["arms"]["positions"],self.taskthree_arm_values[self.taskthree_pose_state])):
                self.is_busy = False
            else:
                pass
        
        else:
            for i in range(7):
                action["arms"]["joint_values"][i*2] = msg.joint_q[i+12]    # 左臂
                action["arms"]["joint_values"][i*2+1] = msg.joint_q[i+19]  # 右臂

        self.current_action = action

    def next_action(self, obs):
        if rospy.is_shutdown() or (obs["extras"].get("info") not in ["TaskOne is done","TaskTwo is done","TaskThree is done",None,"Time limit reached"]):
            self.write_log_to_file(obs["extras"].get("info"))
            self.cleanup()
        
        self.current_obs = obs
        self.last_obs = obs
        self.process_obs(obs)

        st = time.time()

        # 根据任务ID执行不同的任务
        if obs["extras"].get("Current_Task_ID") == "TaskOne":
            self.taskone(obs)
        else:
            self.tasktwo(obs)

        while self.current_action is None and not rospy.is_shutdown():
            # 发布传感器数据
            self.process_obs(self.last_obs, republish=True)
            # 等待一个发布周期
            self.publish_rate.sleep()

        freq = Float32()
        freq.data = 1
        self.freq_pub.publish(freq) 
        return self.current_action # action 
    
    def taskone(self, obs) -> None:
        """任务一:爬楼梯→穿越沙滩→到达终点"""
        current_pos = obs["Kuavo"]["body_state"]["world_position"]
        current_quat = obs["Kuavo"]["body_state"]["world_orient"]
        
        if self.start_x is None:
            self.start_x = current_pos[0]
        
        if TaskSolver.taskone_state == 0:  # 前往楼梯
            if self.walk_to_point_task1(current_quat, self.TARGET_QUAT, current_pos, 
                                 self.TASKONE_START_STAIR_POSITION, 0.3):
                TaskSolver.taskone_state = 1
                
        elif TaskSolver.taskone_state == 1:  # 爬楼梯
            if TaskSolver.taskone_API:
                self.start_stair_climb()
                self.stair_climb_start_time = time.time()
                TaskSolver.taskone_API = False
                
            # 检查是否完成爬楼梯
            height_reached = abs(current_pos[2] - self.TASKONE_END_STAIR_POSITION[2]) <= self.EPS
            time_elapsed = time.time() - self.stair_climb_start_time
            
            if (not self.is_busy and height_reached and time_elapsed > 40) or time_elapsed > 50:
                TaskSolver.taskone_state = 2
                
        elif TaskSolver.taskone_state == 2:  # 前往沙滩初始化点
            if self.walk_to_point_task1(current_quat, self.TARGET_QUAT, current_pos,
                                 self.TASKONE_INITIAL_BEACH_POSITION, 0.1):
                TaskSolver.taskone_state = 3

        elif TaskSolver.taskone_state == 3:  # 前往沙滩起点
            if self.walk_to_point_task1(current_quat, self.TARGET_QUAT, current_pos,
                                 self.TASKONE_START_BEACH_POSITION, 0.1):
                TaskSolver.taskone_state = 4
        
        elif TaskSolver.taskone_state == 4:  # 前往沙滩终点
            if self.walk_to_point_task1(current_quat, self.TARGET_QUAT, current_pos,
                                 self.TASKONE_END_BEACH_POSITION, 0.1):
                TaskSolver.taskone_state = 5
                
        elif TaskSolver.taskone_state == 5:  # 前往最终位置
            self.walk_to_point_task1(current_quat, self.TASKONE_END_QUAT, current_pos, 
                              self.TASKONE_END_POSITION, 0.1, True)
    
    def _monitor_stair_process(self):
        """监控爬楼梯进程状态"""
        # 等待进程结束
        self.stair_process.wait()
        
        # 获取返回码
        return_code = self.stair_process.returncode
        
        # 根据返回码处理结果
        if return_code == 0:
            rospy.loginfo("Stair climbing completed successfully")
        else:
            rospy.logerr(f"Stair climbing failed with return code {return_code}")
        
        # 重置状态
        self.is_busy = False
        self.current_action_name = None
        self.stair_process = None

    def start_stair_climb(self) -> None:
        """启动爬楼梯规划器"""
        self.is_busy = True
        self.current_action_name = "stair_climbing"
        
        command = f"env -i bash -c 'source {CONTORLLER_PATH}/devel/setup.bash && rosrun humanoid_controllers stairClimbPlanner.py'"
        print(command)
        
        try:
            self.stair_process = subprocess.Popen(
                command,
                shell=True,
                stdout=None,
                stderr=None,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            rospy.loginfo(f"Successfully started stair climbing planner")
            
        except Exception as e:
            rospy.logerr(f"Failed to start stair climbing planner: {str(e)}")
            if self.stair_process is not None:
                try:
                    os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
                except:
                    pass
                self.stair_process = None
        
        # 启动监控线程
        monitor_thread = threading.Thread(target=self._monitor_stair_process)
        monitor_thread.daemon = True
        monitor_thread.start()
    
    def walk_to_point_task1(self, current_quat, target_quat, current_pos, target_pos, speed, flag=False) -> bool:
        """任务一使用的走路方法，在到达目标时更新状态"""
        if self.is_busy:
            rospy.logwarn("Cannot walk during busy action")
            return False

        # 计算朝向差
        r_current = Rotation.from_quat([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
        r_target = Rotation.from_quat([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        current_yaw = r_current.as_euler('xyz')[2]
        target_yaw = r_target.as_euler('xyz')[2]

        yaw_diff = target_yaw - current_yaw
        if yaw_diff > np.pi: 
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi: 
            yaw_diff += 2 * np.pi
        
        # 计算距离
        current_pos_arr = np.array(current_pos)
        distance_to_target = np.linalg.norm(current_pos_arr[:2] - target_pos[:2])
        
        if abs(yaw_diff) > 0.05:
            # 需要转向
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            if yaw_diff > 0:
                self.cmd_pose_msg.angular.z = -0.2
            else:
                self.cmd_pose_msg.angular.z = 0.2
        elif distance_to_target > 0.1:
            # 移动
            self.cmd_pose_msg.angular.z = 0.0
            y_diff = target_pos[1] - current_pos[1]
            x_diff = target_pos[0] - current_pos[0]
            
            if abs(x_diff) > 0.05:
                # 沿x轴移动
                if x_diff > 0:
                    # 目标在正x方向，向右移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = -speed
                else:
                    # 目标在负x方向，向左移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = speed
            elif abs(y_diff) > 0.05:
                # 沿y轴移动
                if y_diff > 0:
                    # 目标在正y方向，向前移动
                    self.cmd_pose_msg.linear.x = speed
                    self.cmd_pose_msg.linear.y = 0.0
                else:
                    # 目标在负y方向，向后移动
                    self.cmd_pose_msg.linear.x = -speed
                    self.cmd_pose_msg.linear.y = 0.0
            else:
                # 接近目标，减速
                self.cmd_pose_msg.linear.x = 0.1
                self.cmd_pose_msg.linear.y = 0.1
        else:
            # 到达目标位置
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0
            self.cmd_pose_pub.publish(self.cmd_pose_msg)
            return True
        
        if flag:
            self.cmd_pose_msg.linear.x, self.cmd_pose_msg.linear.y = self.cmd_pose_msg.linear.y, self.cmd_pose_msg.linear.x
        
        self.cmd_pose_pub.publish(self.cmd_pose_msg)
        return False

    def tasktwo(self, obs) -> None:

        if(self.is_busy): 
            return 

        current_pos = obs["Kuavo"]["body_state"]["world_position"] 
        current_quat = obs["Kuavo"]["body_state"]["world_orient"]

        if self.is_tasktwo_ready or len(self.placed_objects) == 3:  # if len(self.placed_objects) == len(self.target_classes): 
            if self.is_tasktwo_ready or self.walk_to_point_task2(current_quat,
                                                                  self.ideal_dir_values_task2_grabbing(self.dest),
                                                                  current_pos,
                                                                  self.ITEM_POS[self.dest],
                                                                  1.0,
                                                                  self.ideal_dir_str_task2_grabbing(self.dest)):
                self.is_tasktwo_ready = True
                self.tasktwo_state = 1   # task 3
        else:
            self.tasktwo_state = 0

        if(self.tasktwo_state == 0):
            # 移动到目标位置
            grabbing_dir = self.ideal_dir_values_task2_grabbing(self.dest)
            target_pos = self.ITEM_POS[self.dest]
            grabbing_direction = self.ideal_dir_str_task2_grabbing(self.dest) # str

            if self.is_examing or self.walk_to_point_task2(current_quat, grabbing_dir, 
                                        current_pos, target_pos, 2, grabbing_direction):
                if self.stage == "place" and self.dest == 13:  # 箱子位置
                    self.is_examing = True
                    self.write_log_to_file("到达place位置")
                    self.place_object(obs)

                elif self.stage == "grab" and self.dest not in self.empty_places:  # 抓取位置
                    if self.is_examing or self.examine_object(obs):
                        self.is_examing = True

                        if(self.walk_to_point_task2(current_quat, grabbing_dir,
                                                current_pos,self.GRAB_POS[self.dest],1,grabbing_direction)):
                                self.write_log_to_file("到达抓取位置")
                                self.grab_object(obs) 
                        else:
                                self.write_log_to_file("尝试前往抓取位置")
                    else:
                        self.dest = (self.dest % 15) + 1  
                # 更新下一个目标位置
                else:
                    self.dest = (self.dest % 15) + 1

        # ------------------------------------------------
        # task 3
        elif(self.tasktwo_state == 1):
            
            if(self.walk_to_point(current_quat,
                                  self.ideal_dir_values_task3(self.taskthree_dest),
                                  current_pos,
                                  self.item_pos[self.taskthree_dest],
                                  1.0,
                                  self.ideal_dir_str_task3(self.taskthree_dest))):
                
                if(self.taskthree_dest == 14):
                    self.taskthree_dest = 13
                elif(self.taskthree_dest == 13):
                    self.is_busy = True # grab
                    self.taskthree_dest = 9 

                elif(self.taskthree_dest == 9):
                    self.taskthree_dest = 8

                elif(self.taskthree_dest == 8):
                    self.taskthree_dest = 16

                elif(self.taskthree_dest == 16):
                    self.taskthree_dest = 17

                elif(self.taskthree_dest == 17):
                    self.taskthree_arrive = True
                    # end of task 3
                else:
                    pass
#       ---------------------------------------------------
        else: # 这个状态专门用于试验
            if(self.walk_to_point(current_quat,
            self.dir_quat[3],
            current_pos,
            [4,current_pos[1],current_pos[2]],
            1.0,
            "west")): 
                self.taskthree_dest = 16
                pass
        # ------------------------------------------------

    def walk_to_point_task2(self,current_quat,target_quat,current_pos,target_pos,speed,dir = "north") ->bool:
        if(self.is_busy): 
            return False
        # yaw
        r_current = Rotation.from_quat([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
        r_target = Rotation.from_quat([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        current_yaw = r_current.as_euler('xyz')[2]
        target_yaw = r_target.as_euler('xyz')[2]

        yaw_diff = target_yaw - current_yaw
        if yaw_diff > np.pi: 
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi: 
            yaw_diff += 2 * np.pi
        current_pos_arr = np.array(current_pos)
        distance_to_target = np.linalg.norm(current_pos_arr[:2] - target_pos[:2])
        if abs(yaw_diff) > 0.05:
        # 需要转向
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            if yaw_diff > 0:
                self.cmd_pose_msg.angular.z = -0.2
            else:
                self.cmd_pose_msg.angular.z = 0.2
        elif distance_to_target > 0.1:
            self.cmd_pose_msg.angular.z = 0.0
            # 目标方向是y轴正方向，所以主要沿y轴移动
            y_diff = target_pos[1] - current_pos[1]
            x_diff = target_pos[0] - current_pos[0]
        # 根据位置差决定移动方向
            if abs(x_diff) > 0.05:
                    # 沿x轴移动
                if x_diff > 0:
                        # 目标在正x方向，向右移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = -speed
                else:
                        # 目标在负x方向，向左移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = speed
            elif abs(y_diff) > 0.05:
                    # 沿y轴移动
                if y_diff > 0:
                        # 目标在正y方向，向前移动
                    self.cmd_pose_msg.linear.x = speed
                    self.cmd_pose_msg.linear.y = 0.0
                else:
                        # 目标在负y方向，向后移动
                    self.cmd_pose_msg.linear.x = -speed
                    self.cmd_pose_msg.linear.y = 0.0
            else:
                    # 接近目标，减速
                self.cmd_pose_msg.linear.x = 0.1
                self.cmd_pose_msg.linear.y = 0.1
        else: # 到达目标位置
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0
            self.cmd_pose_pub.publish(self.cmd_pose_msg)  
            return True
        
        if(dir == "west"): # west 
            self.cmd_pose_msg.linear.x,self.cmd_pose_msg.linear.y = self.cmd_pose_msg.linear.y, -self.cmd_pose_msg.linear.x
        elif(dir == "south"):
            self.cmd_pose_msg.linear.x *= -1
            self.cmd_pose_msg.linear.y *= -1
        elif(dir == "east"):
            self.cmd_pose_msg.linear.x,self.cmd_pose_msg.linear.y = -self.cmd_pose_msg.linear.y,self.cmd_pose_msg.linear.x
        else: # north
            pass
        self.cmd_pose_pub.publish(self.cmd_pose_msg)  
        return False
    #   支持沿坐标轴的走路的算法

    def ideal_dir_str_task2_grabbing(self, dest):
        if dest in range(1, 4):
            return "west"
        elif dest in range(4, 9):
            return "north"
        elif dest in range(9, 10) or dest in range(13, 15): 
            return "east"
        else:
            return "south"

    def ideal_dir_values_task2_grabbing(self, dest):
        if dest in range(1, 4):
            return self.DIR_QUAT[3]  # west
        elif dest in range(4, 9):
            return self.DIR_QUAT[1]  # north
        elif dest in range(9, 10) or dest in range(13, 15): 
            return self.DIR_QUAT[4]  # east
        else:
            return self.DIR_QUAT[2]  # south

    def ideal_dir_values_task2(self,dest): # only in task2  
        if(dest in range(1,4)):
            return self.dir_quat[3] # west
        elif(dest in range(4,9)):
            return self.dir_quat[1] # north
        elif(dest in range(9,10) or dest in range(13,15)): 
            return self.dir_quat[4] # east
        else:
            return self.dir_quat[2] # south 
    
    def ideal_dir_str_task2(self,dest): # only in task2
        if(dest in range(1,4)):
            return "west"
        elif(dest in range(4,9)):
            return "north"
        elif(dest in range(9,10) or dest in range(13,15)): 
            return "east"
        else:
            return "south"
        
    def ideal_dir_values_task3(self,dest): # only in task3
        if(dest == 14 or dest == 16):
            return self.dir_quat[3] # west
        elif(dest == 13 or dest == 9):
            return self.dir_quat[4] # east
        elif(dest == 17 or dest == 8): 
            return self.dir_quat[2] # south
        else:   
            return self.dir_quat[1] # north
    
    def ideal_dir_str_task3(self,dest): # only in task3
        if(dest == 14 or dest == 16):
            return "west"
        elif(dest == 13 or dest == 9):
            return "east"
        elif(dest == 17 or dest == 8):
            return "south"
        else:
            return "north"
        
    def detect_objects(self, obs) -> list:
        """使用YOLOv8模型检测物品，优先保留未抓取的目标"""
        self.detected_objects = []   # 重置为空的列表
        
        # 检查是否有相机数据
        if not obs.get("camera") or obs["camera"]["rgb"] is None:
            rospy.logwarn("未获取到相机RGB数据")
            return self.detected_objects
        
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
                least_depth = 0.7 #只保留深度大于0.7m的检测结果

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
                            u = (box[0] + box[2]) / 2
                            v = (box[1] + box[3]) / 2
                            
                            # 获取深度信息
                            depth = self.get_depth_value(obs, u, v)
                            if depth is None:
                                continue

                            
                            # 直接记录深度值，而不是转换到世界坐标
                            if depth > least_depth:
                                self.detected_objects.append({
                                    "class_name": class_name,
                                    "depth": depth,  # 相机坐标系下的深度值
                                    "u": u,          # 图像坐标u
                                    "v": v,          # 图像坐标v
                                    "confidence": confidence
                                })
                                rospy.loginfo(f"检测到: {class_name} @ 深度={depth:.3f}m, 置信度: {confidence:.2f}")
                return self.detected_objects
            except Exception as e:
                rospy.logerr(f"物体检测失败: {str(e)}")
                return self.detected_objects
        else:
            rospy.logwarn("YOLO模型未加载，无法检测物体")
            return self.detected_objects
    
    def examine_object(self, obs) -> bool:
        """
        判断: 当前检测中最近物体是否属于目标类别 self.target_class_name
        返回 True/False；只在 True 时记录 locked_target；不修改 target_class_name。
        """
        logs = ["==== examine_object 最近物体判定 ===="]

        objs = self.detect_objects(obs)
        if not objs:
            logs.append("无检测 -> False")
            self.print_in_new_terminal("\n".join(logs), "examine_object")
            return False

        target_class = self.target_class_name  # 仅读
        logs.append(f"目标类别: {target_class if target_class else '未设置(恒 False)'}")
        logs.append(f"检测数量: {len(objs)}")

        for o in objs:
            logs.append(f"- {o['class_name']} conf={o['confidence']:.2f} "
                        f"depth={o['depth']:.2f}")

        nearest = min(objs, key=lambda x: x["depth"])  # 仍用深度或改用 distance_h
        logs.append(f"最近物体: {nearest['class_name']} depth={nearest['depth']:.2f}")

        if not target_class:
            logs.append("无目标类别 -> False")
            self.print_in_new_terminal("\n".join(logs), "examine_object")
            return False

        if nearest["class_name"] == target_class:
            logs.append("匹配成功 -> True (locked_target 已更新)")
            self.print_in_new_terminal("\n".join(logs), "examine_object")
            return True
        else:
            logs.append("匹配失败 -> False")
            self.print_in_new_terminal("\n".join(logs), "examine_object")
            return False

    def get_depth_value(self, obs, x, y) -> Optional[float]:
        """获取深度图中指定坐标的深度值"""
        if not obs.get("camera") or obs["camera"]["depth"] is None:
            rospy.logwarn("未获取到深度图数据")
            return None
        
        depth_map = obs["camera"]["depth"]
        height, width = depth_map.shape[:2]
        
        # 确保坐标在图像范围内
        x = int(np.clip(x, 0, width-1))
        y = int(np.clip(y, 0, height-1))
        
        # 获取深度值
        depth_value = depth_map[y, x]
        
        # 检查深度值是否有效
        if depth_value <= 0 or depth_value > 10:
            rospy.logwarn(f"无效深度值: {depth_value} @ ({x}, {y})")
            return None
        
        return depth_value
    
    def print_in_new_terminal(self, message, title="Object Detection"):
            """在新终端窗口中显示消息"""
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
        """检查命令是否可用"""
        try:
            subprocess.check_call(f"which {command}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError:
            return False
        
    def grab_object(self, obs) -> None:
        """
        抓取逻辑:
        - 使用 locked_target (若为空则回退目标类别字符串)
        - 成功抓取后记录类别到 grabbed_objects
        """
        # 解析当前要抓的类别
        target_class = self.target_class_name

        if self.ifpick:  # 动作已完成
            self.grabbed_objects.append(target_class)
            self.write_log_to_file(f"抓取成功: {target_class}")
            # 进入放置阶段
            self.stage = "place"
            self.is_grabbing_obj = False
            self.grab = 0
            self.is_examing = False
            self.empty_places.append(self.dest)
            self.dest = (self.dest % 15) + 1
        else:
            # 触发抓取动作序列
            self.is_grabbing_obj = True
            self.is_busy = True
            self.grab = 1

    def place_object(self, obs) -> None:
        """
        放置逻辑:
        - 放下 locked_target / target_class_name
        - 完成后清空 locked_target
        - 选择下一目标类别(若还有未处理)
        """
        placed_class = self.target_class_name

        if self.ifpick is False:  # 手已松开表示放置完成
            self.placed_objects.append(placed_class)
            self.write_log_to_file(f"放置完成: {placed_class}")
            self.stage = "grab"
            self.is_placing = False
            self.is_examing = False
            self.dest = (self.dest % 15) + 1
        else:
            # 正在举手放置过程
            self.is_busy = True
            self.is_placing = True  

    def arm_pos_cmp(self,l1,l2) -> bool:
        for i in range(len(l1)):
            if(abs(l1[i] - l2[i]) > 0.04): # stricter 
                return False
        return True       

    def write_log_to_file(self,infomation):
        # 获取当前时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # 创建日志内容
        log_content = f"{infomation} + {current_time}"
        # 定义文件名（包含日期避免覆盖）
        filename = f"running_log{datetime.datetime.now().strftime('%Y%m%d')}.txt"
        # 写入文件（追加模式）
        with open(filename, "a", encoding="utf-8") as log_file:
            log_file.write(log_content + "\n\n")
            rospy.loginfo(f"日志已写入文件: {filename}")              

    def image_to_world(self, u, v, depth, obs) -> Optional[np.ndarray]:
        """将图像中的一个像素点精确地映射到三维世界中的具体位置
            使机器人能够理解和操作物理世界中的物体"""
        # 相机内参
        fx = self.camera_intrinsics[0, 0] # x轴焦距
        fy = self.camera_intrinsics[1, 1] # y轴焦距
        cx = self.camera_intrinsics[0, 2] # 主点x坐标
        cy = self.camera_intrinsics[1, 2] # 主点y坐标
        
        # 使用小孔相机模型将2D像素坐标转换为3D相机坐标
        z_cam = depth
        x_cam = (u - cx) * z_cam / fx
        y_cam = (v - cy) * z_cam / fy
        
        # 相机在机器人坐标系中的位置
        # --------------------------------------------
        camera_pos_in_robot = np.array([0.0, 0.0, 1])  
        # --------------------------------------------
        # 假设相机高度1米
        
        # 相机坐标系到机器人坐标系的旋转（假设相机朝前）
        # 旋转矩阵（单位矩阵表示无旋转）
        R_cam_to_robot = np.eye(3)
        
        # 相机坐标系下的点转换到机器人坐标系
        point_in_robot = R_cam_to_robot @ np.array([x_cam, y_cam, z_cam]) + camera_pos_in_robot
        
        # 获取机器人位姿
        robot_pos = np.array(obs["Kuavo"]["body_state"]["world_position"])
        robot_quat = obs["Kuavo"]["body_state"]["world_orient"]
        
        # 转换为旋转矩阵
        r = Rotation.from_quat([
            robot_quat[3],  # w
            robot_quat[0],  # x
            robot_quat[1],  # y
            robot_quat[2]   # z
        ])
        R_robot_to_world = r.as_matrix()
        
        # 机器人坐标系到世界坐标系
        point_in_world = R_robot_to_world @ point_in_robot + robot_pos
        
        return point_in_world
    
    def select_next_target(self) -> None:
        """从识别结果中选择下一个未抓取的目标"""
        #  按置信度排序
        self.detected_objects.sort(key=lambda x: x["confidence"], reverse=True)
        
        for obj in self.detected_objects:
            if obj["class_name"] not in self.grabbed_objects:
                self.current_target = obj
                rospy.loginfo(f"选中目标: {obj['class_name']}, 置信度: {obj['confidence']:.2f}")
                return
        
        # 如果没找到新目标
        self.current_target = None
        rospy.loginfo("未找到新目标，等待重新识别...")

    def walk_to_point(self,current_quat,target_quat,current_pos,target_pos,speed,dir = "north") ->bool:

        if(self.is_busy): 
            return # not avaliable 

        # yaw
        r_current = Rotation.from_quat([current_quat[3], current_quat[0], current_quat[1], current_quat[2]])
        r_target = Rotation.from_quat([target_quat[3], target_quat[0], target_quat[1], target_quat[2]])
        current_yaw = r_current.as_euler('xyz')[2]
        target_yaw = r_target.as_euler('xyz')[2]

        yaw_diff = target_yaw - current_yaw
        if yaw_diff > np.pi: 
            yaw_diff -= 2 * np.pi
        elif yaw_diff < -np.pi: 
            yaw_diff += 2 * np.pi
        current_pos_arr = np.array(current_pos)
        distance_to_target = np.linalg.norm(current_pos_arr[:2] - target_pos[:2])
        if abs(yaw_diff) > 0.05:
        # 需要转向
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            if yaw_diff > 0:
                self.cmd_pose_msg.angular.z = -0.4
            else:
                self.cmd_pose_msg.angular.z = 0.4
        elif distance_to_target > 0.1:
            self.cmd_pose_msg.angular.z = 0.0
            # 目标方向是y轴正方向，所以主要沿y轴移动
            y_diff = target_pos[1] - current_pos[1]
            x_diff = target_pos[0] - current_pos[0]
        # 根据位置差决定移动方向
            if abs(x_diff) > 0.05:
                    # 沿x轴移动
                if x_diff > 0:
                        # 目标在正x方向，向右移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = -speed
                else:
                        # 目标在负x方向，向左移动
                    self.cmd_pose_msg.linear.x = 0.0
                    self.cmd_pose_msg.linear.y = speed
            elif abs(y_diff) > 0.05:
                    # 沿y轴移动
                if y_diff > 0:
                        # 目标在正y方向，向前移动
                    self.cmd_pose_msg.linear.x = speed
                    self.cmd_pose_msg.linear.y = 0.0
                else:
                        # 目标在负y方向，向后移动
                    self.cmd_pose_msg.linear.x = -speed
                    self.cmd_pose_msg.linear.y = 0.0
            else:
                    # 接近目标，减速
                self.cmd_pose_msg.linear.x = 0.4
                self.cmd_pose_msg.linear.y = 0.4

        else: # 到达目标位置
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0
            self.cmd_pose_pub.publish(self.cmd_pose_msg)  
            return True
        
        if(dir == "west"): # west 
            self.cmd_pose_msg.linear.x,self.cmd_pose_msg.linear.y = self.cmd_pose_msg.linear.y, -self.cmd_pose_msg.linear.x
        elif(dir == "south"):
            self.cmd_pose_msg.linear.x *= -1
            self.cmd_pose_msg.linear.y *= -1
        elif(dir == "east"):
            self.cmd_pose_msg.linear.x,self.cmd_pose_msg.linear.y = -self.cmd_pose_msg.linear.y,self.cmd_pose_msg.linear.x
        else: # north
            pass
        self.cmd_pose_pub.publish(self.cmd_pose_msg)  
        return False
    #   支持沿坐标轴的走路的算法
    # ----------------------------------------

    def process_obs(self, obs: Dict[str, Any], republish=False) -> None:
        """处理观测数据并发布传感器数据
            self.sensor_pub.publish(sensor_data)
        Args:
            obs: 观测数据字典，包含IMU和关节状态信息
        """
        sensor_data = sensorsData()
        
        # 设置时间戳
        current_time = rospy.Time.now()
        sensor_data.header.stamp = current_time
        sensor_data.header.frame_id = "world"
        sensor_data.sensor_time = rospy.Duration(self.sensor_time)
        if republish:
            pass
        else:
            self.sensor_time += obs["imu_data"]["imu_time"] - self.last_sensor_time
        self.last_sensor_time = obs["imu_data"]["imu_time"]

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
            sensor_data.joint_data.joint_q = [0.0] * 28
            sensor_data.joint_data.joint_v = [0.0] * 28
            sensor_data.joint_data.joint_vd = [0.0] * 28
            sensor_data.joint_data.joint_torque = [0.0] * 28

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

    def init_tasktwo_robot_pos(self): 
    # only needed to apply once
    # 让机器人在一开始就保持一个适合取物品的姿势，就决定是举起的姿势了
        self.raise_arm()
        self.force_arm_pos()

    def return_to_idle(self): # 机器人回到站立，双手自然下垂
        # only needed to apply once

        self.arm_pose_msg.times = self.adjust_times
        self.arm_pose_msg.values = self.idle_arm_values

        starttime = time.time()
        self.arm_pose_pub.publish(self.arm_pose_msg)

        while(time.time() - starttime >= self.adjust_times[0] + 5):
            pass # 硬等

        # still 
        starttime = time.time()
        while(time.time() - starttime >= 0.5): # 250 times
            self.cmd_pose_msg.linear.x = 0.0
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0 
            self.cmd_pose_pub.publish(self.cmd_pose_msg) 
            # 持续发布停稳的消息
        
        self.release_arm_pos()
        return

    # 我现在准备写方法：操控机器人沿着肩关节在竖直平面内旋转让手掌升高和降低
    # only needed to apply once
    def adjust_arm(self,arm_values) -> None:

        """
        self.arm_pose_msg.times = self.adjust_times
        self.arm_pose_msg.values = arm_values
        # starttime = time.time()
        self.arm_pose_pub.publish(self.arm_pose_msg)

        rospy.sleep(self.adjust_times[0] + 1)
        """
        self.arm_traj_change(5,arm_values) # 
        self.write_log_to_file(str(arm_values))
    
    # only needed to apply once
    def raise_arm(self):
        self.adjust_arm(self.lift_values)
    # only needed to apply once
    def lower_arm(self): 
    # 准备拿取物品的姿势，可以用于给机器人初始化姿势
        self.adjust_arm(self.lower_values)
    # only needed to apply once
    def force_arm_pos(self) -> bool: # 强行手部动作保持
        self.request.control_mode = 0
        response = self.arm_traj_change_mode_client(self.request)
        return response.result
    # only needed to apply once
    def release_arm_pos(self) -> bool: # 允许保持手部自然摆动
        self.request.control_mode = 1
        response = self.arm_traj_change_mode_client(self.request)
        return response.result

    def carry_box(self): 
        self.adjust_arm([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0])
        self.adjust_arm([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0])
        self.adjust_arm([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0])
        self.adjust_arm([-20,-20,0,0,-5,5,-50,-50,0,0,0,0,0,0])
        rospy.sleep(1)
        # self.move_back_severalsteps()  # carried 
        self.write_log_to_file("box carried")

    def place_box(self):
        self.adjust_arm([-20,-20,0,0,0,0,-70,-70,0,0,0,0,0,0])
        self.adjust_arm([0,0,0,0,0,0,-100,-100,0,0,0,0,0,0])
        self.adjust_arm([0,0,0,0,90,-90,-100,-100,0,0,0,0,0,0])
        self.adjust_arm(self.idle_arm_values)
        rospy.sleep(1)
        # self.move_back_severalsteps()
        self.write_log_to_file("box placed")

    def move_back_severalsteps(self):
        starttime = time.time()
        # ----------------------------------------------
        while(time.time() - starttime <=2):
            self.cmd_pose_msg.linear.x = -0.8
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0 
            self.cmd_pose_pub.publish(self.cmd_pose_msg) 
            # 持续发布back的消息

    def move_forward_severalsteps(self):
        starttime = time.time()
        # ----------------------------------------------
        while(time.time() - starttime <=2):
            self.cmd_pose_msg.linear.x = 0.8
            self.cmd_pose_msg.linear.y = 0.0
            self.cmd_pose_msg.angular.z = 0.0 
            self.cmd_pose_pub.publish(self.cmd_pose_msg) 
            # 持续发布forward的消息

    def write_log_to_file(self,infomation):
    # 获取当前时间并格式化为字符串
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 创建日志内容
        log_content = f"{infomation} + {current_time}"
    # 定义文件名（包含日期避免覆盖）
        filename = f"running_log{datetime.datetime.now().strftime('%Y%m%d')}.txt"
    # 写入文件（追加模式）
        with open(filename, "a", encoding="utf-8") as log_file:
            log_file.write(log_content + "\n\n")
            rospy.loginfo(f"日志已写入文件: {filename}")

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self.launch_process is not None:
            try:
                rospy.loginfo("Cleaning up launch process...")
                os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                self.launch_process.wait()
                self.launch_process = None
                rospy.loginfo("Launch process cleaned up")
            except Exception as e:
                rospy.logerr(f"Error cleaning up launch process: {str(e)}")
            
            # 清理爬楼梯进程
            if hasattr(self, 'stair_process') and self.stair_process is not None:
                try:
                    rospy.loginfo("Cleaning up stair climbing process...")
                    os.killpg(os.getpgid(self.stair_process.pid), signal.SIGTERM)
                    self.stair_process.wait()
                    self.stair_process = None
                    rospy.loginfo("Stair climbing process cleaned up")
                except Exception as e:
                    rospy.logerr(f"Error cleaning up stair climbing process: {str(e)}")

            # 清理抓箱子进程
            if hasattr(self, 'grab_box_process') and self.grab_box_process is not None:
                try:
                    rospy.loginfo("Cleaning up grab box process...")
                    os.killpg(os.getpgid(self.grab_box_process.pid), signal.SIGTERM)
                    self.grab_box_process.wait()
                    self.grab_box_process = None
                    rospy.loginfo("Grab box process cleaned up")
                except Exception as e:
                    rospy.logerr(f"Error cleaning up grab box process: {str(e)}")

    def arm_pos_cmp(self,l1,l2) -> bool:
        for i in range(len(l1)):
            if(abs(l1[i] - l2[i]) > 0.1):
                return False
        return True

    def arm_traj_change(self, duration: float, target_positions: list) -> None:

        """平滑调整手臂到目标位置
        Args:
            duration: 调整时间（秒）
            target_positions: 目标位置列表（角度值）
        """
        rospy.loginfo(f"开始手臂轨迹调整，持续时间: {duration}秒")
        
        # 转换目标位置从角度到弧度
        target_positions_rad = [angle * np.pi / 180.0 for angle in target_positions]
        
        # 确保目标位置长度正确
        if len(target_positions_rad) != len(self.current_arms_position):
            rospy.logwarn(f"目标位置长度({len(target_positions_rad)})与手臂关节数({len(self.current_arms_position)})不匹配")
            # 裁剪或填充目标位置
            if len(target_positions_rad) > len(self.current_arms_position):
                target_positions_rad = target_positions_rad[:len(self.current_arms_position)]
            else:
                target_positions_rad += [0.0] * (len(self.current_arms_position) - len(target_positions_rad))
        
        # 计算总共需要的步数
        total_steps = int(duration * self.control_freq)
        
        # 计算每步的增量
        increments = []
        for i in range(len(self.current_arms_position)):
            start = self.current_arms_position[i]
            end = target_positions_rad[i]
            increments.append((end - start) / total_steps)
        
        # 逐步调整位置
        for step in range(total_steps):
            # 更新当前位置
            for i in range(len(self.current_arms_position)):
                self.current_arms_position[i] += increments[i]
                if self.current_action and 'arms' in self.current_action:
                    # 确保只更新前14个关节值（手臂部分）
                    self.current_action['arms']['joint_values'] = self.current_arms_position[:14]
            
            # 等待一个控制周期
            self.publish_rate.sleep()
        
        # 确保精确到达目标位置
        self.current_arms_position = target_positions_rad
        if self.current_action and 'arms' in self.current_action:
        # 确保只更新前14个关节值（手臂部分）
            self.current_action['arms']['joint_values'] = self.current_arms_position[:14]

        self.write_log_to_file(str(self.current_arms_position))