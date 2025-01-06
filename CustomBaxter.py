import rospy
import cv_bridge
from baxter_core_msgs.msg import JointCommand, EndpointState, CameraSettings
from baxter_core_msgs.msg import EndEffectorCommand, EndEffectorProperties, EndEffectorState,DigitalIOState
from baxter_core_msgs.srv import OpenCamera, CloseCamera, SolvePositionIK, SolvePositionIKRequest
from std_msgs.msg import Bool, Header
from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped
from sensor_msgs.msg import JointState, Image
from sensor_msgs.msg import Range
import cv2
from enum import Enum

class GripperAction(Enum):
    CALIBRATE = 0
    GRIP = 1
    RELEASE = 2

class CustomBaxter:
    def __init__(self):
        self.left_joint_names = [f"left_{x}" for x in ["e0", "e1", "s0", "s1",  "w0", "w1", "w2"]]
        self.right_joint_names = [f"right_{x}" for x in ["e0", "e1", "s0", "s1",  "w0", "w1", "w2"]]

        self.left_joint_angles = {}
        self.right_joint_angles = {}
        self.left_gripper_state = EndEffectorState()
        self.right_gripper_state = EndEffectorState()

        self.record_button_state = False   
        self.record_button_prev_state = 0

        self.right_record_button_state = False   
        self.right_record_button_prev_state = 0

        self.close_button = False

        self.cam_image = Image()

        self.cam_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.fill_cam_image, queue_size=1)

        self.joint_state_sub = rospy.Subscriber("/robot/joint_states", JointState, self.fill_joint_state)
    

        self.sub_left_gripper_state = rospy.Subscriber("/robot/end_effector/left_gripper/state", EndEffectorState, self.fill_left_gripper_state)
        self.sub_right_gripper_state = rospy.Subscriber("/robot/end_effector/right_gripper/state", EndEffectorState, self.fill_right_gripper_state)

        rospy.Subscriber("/robot/digital_io/left_button_ok/state", DigitalIOState, self.record_pressed)
        rospy.Subscriber("/robot/digital_io/right_button_ok/state", DigitalIOState, self.right_record_pressed)

        rospy.Subscriber("/robot/digital_io/left_lower_button/state", DigitalIOState, self.left_release_pressed)
        rospy.Subscriber("/robot/digital_io/right_lower_button/state", DigitalIOState, self.right_release_pressed)

        rospy.Subscriber("/robot/digital_io/left_upper_button/state", DigitalIOState, self.left_grip_pressed)
        rospy.Subscriber("/robot/digital_io/right_upper_button/state", DigitalIOState, self.right_grip_pressed)

        self.pub_left_joint_cmd = rospy.Publisher("/robot/limb/left/joint_command", JointCommand, queue_size=1)
        self.pub_right_joint_cmd = rospy.Publisher("/robot/limb/right/joint_command", JointCommand, queue_size=1)

        self.left_gripper_pub = rospy.Publisher(f'/robot/end_effector/left_gripper/command', EndEffectorCommand, queue_size=10)
        self.right_gripper_pub = rospy.Publisher(f'/robot/end_effector/right_gripper/command', EndEffectorCommand, queue_size=10)

        self.robot_joint_enable = rospy.Publisher('/robot/set_super_enable', Bool, queue_size=10)

    def fill_cam_image(self, msg):
        self.cam_image = msg    

    def fill_joint_state(self, msg):
        for idx, name in enumerate(msg.name):
            if name in self.left_joint_names:
                self.left_joint_angles[name] = msg.position[idx]
            if name in self.right_joint_names:
                self.right_joint_angles[name] = msg.position[idx]

    def fill_left_gripper_state(self, msg):
        self.left_gripper_state = msg

    def fill_right_gripper_state(self, msg):
        self.right_gripper_state = msg     

    def set_left_joint_angles(self, angles):
        left_joint_command = JointCommand()
        left_joint_command.names = list(angles.keys())
        left_joint_command.command = list(angles.values())
        left_joint_command.mode = JointCommand.POSITION_MODE
        self.pub_left_joint_cmd.publish(left_joint_command)

    def set_right_joint_angles(self, angles):
        right_joint_command = JointCommand()
        right_joint_command.names = list(angles.keys())
        right_joint_command.command = list(angles.values())
        right_joint_command.mode = JointCommand.POSITION_MODE
        self.pub_right_joint_cmd.publish(right_joint_command) 

        
    def set_left_arm_with_tensor(self, tensorlst):
        left_joint_command = JointCommand()
        left_joint_command.names = self.left_joint_names
        left_joint_command.command = tensorlst[0:7]
        left_joint_command.mode = JointCommand.POSITION_MODE
        self.set_left_gripper(tensorlst[7], tensorlst[8])
        self.pub_left_joint_cmd.publish(left_joint_command)

    def set_right_arm_with_tensor(self, tensorlst):
        right_joint_command = JointCommand()
        right_joint_command.names = self.right_joint_names
        right_joint_command.command = tensorlst[0:7]
        right_joint_command.mode = JointCommand.POSITION_MODE
        self.set_right_gripper(tensorlst[7], tensorlst[8])
        self.pub_right_joint_cmd.publish(right_joint_command)  

    def set_right_gripper(self, position, force):
        if not self.right_gripper_state.calibrated:
            rospy.loginfo("Not callibrated")
            self.right_gripper.calibrate()
        if (position < 40):
            self.right_gripper_action(GripperAction.GRIP)
        else:
            self.right_gripper_action(GripperAction.RELEASE)    


    def set_left_gripper(self, position, force):
        if not self.left_gripper_state.calibrated:
            rospy.loginfo("Not callibrated")
            self.left_gripper.calibrate()
        if (position < 40):
            self.left_gripper_action(GripperAction.GRIP)
        else:
            self.left_gripper_action(GripperAction.RELEASE)             
    


    def left_gripper_action(self, action):
        cmd = EndEffectorCommand()
        cmd.id = self.left_gripper_state.id
        action_val = action.value
        if action_val == 0:
            cmd.command = EndEffectorCommand.CMD_CALIBRATE
        elif action_val == 1:
            cmd.command = EndEffectorCommand.CMD_GRIP
        elif action_val == 2:
            cmd.command = EndEffectorCommand.CMD_RELEASE
        self.left_gripper_pub.publish(cmd) 

    def right_gripper_action(self, action):
        cmd = EndEffectorCommand()
        cmd.id = self.right_gripper_state.id
        action_val = action.value
        if action_val == 0:
            cmd.command = EndEffectorCommand.CMD_CALIBRATE
        elif action_val == 1:
            cmd.command = EndEffectorCommand.CMD_GRIP
        elif action_val == 2:
            cmd.command = EndEffectorCommand.CMD_RELEASE
        self.right_gripper_pub.publish(cmd)              

    def enable_robot_joints(self, status):
        self.robot_joint_enable.publish(status)

    def record_pressed(self, msg):
        if msg.state == 1 and self.record_button_prev_state == 0:
            self.record_button_state = not self.record_button_state
            if self.record_button_state:
                rospy.loginfo("Recording started!")
            else:
                rospy.loginfo("Recording stopped!")

        self.record_button_prev_state = msg.state

    def right_record_pressed(self, msg):
        if msg.state == 1 and self.right_record_button_prev_state == 0:
            self.right_record_button_state = not self.right_record_button_state
            if self.record_button_state:
                rospy.loginfo("Recording started!")
            else:
                rospy.loginfo("Recording stopped!")

        self.right_record_button_prev_state = msg.state    

    def disable_pressed(self,msg):
        self.disable_pressed = msg.state

    def get_training_data(self):
        return f"{self.custom_join(self.left_joint_angles.values())};{self.left_gripper_state.position};{self.left_gripper_state.force};{self.custom_join(self.right_joint_angles.values())};{self.right_gripper_state.position};{self.right_gripper_state.force}"    
    
    def get_left_arm_training_data(self):
        return f"{self.custom_join(self.left_joint_angles.values())};{self.left_gripper_state.position};{self.left_gripper_state.force}"
    
    def get_right_arm_training_data(self):
        return f"{self.custom_join(self.right_joint_angles.values())};{self.right_gripper_state.position};{self.right_gripper_state.force}"
    
    def get_left_arm_list(self):
        lst = list(self.left_joint_angles.values())
        lst.append(self.left_gripper_state.position)
        lst.append(self.left_gripper_state.force)
        return lst
    
    def get_right_arm_list(self):
        lst = list(self.right_joint_angles.values())
        lst.append(self.right_gripper_state.position)
        lst.append(self.right_gripper_state.force)
        return lst

    def custom_join(self, lst):
        fstr = ""
        for i in lst:
            fstr += f"{i};"
        return fstr[:-1]

    def publish_from_csv(self, cmd):
        lst = cmd.split(";");    
        left_joint_angles = {name: float(lst[idx]) for idx, name in enumerate(self.left_joint_names)}
    
        right_joint_angles = {name: float(lst[idx]) for idx, name in enumerate(self.right_joint_names, 9)}

 
        self.set_left_gripper(float(lst[7]), float(lst[8]))
        self.set_right_gripper(float(lst[16]), float(lst[17]))
        self.set_left_joint_angles(left_joint_angles)
        self.set_right_joint_angles(right_joint_angles)

    def save_image(self, path):
        bridge = cv_bridge.CvBridge()
        bridged = bridge.imgmsg_to_cv2(img_msg=self.cam_image, desired_encoding="bgr8")
        cv2.imwrite(path,bridged)
    
    def left_release_pressed(self, msg : DigitalIOState):
        if (msg.state):
            self.left_gripper_action(GripperAction.RELEASE)

    def left_grip_pressed(self, msg : DigitalIOState):
        
        if (msg.state):
            self.left_gripper_action(GripperAction.GRIP)
    
    def right_release_pressed(self, msg : DigitalIOState):
        if (msg.state):

            self.right_gripper_action(GripperAction.RELEASE)

    def right_grip_pressed(self, msg : DigitalIOState):
        if (msg.state):
            self.right_gripper_action(GripperAction.GRIP) 


    def ask_for_spoon(self):
        csv_position = "0.0003834951969713534;1.4503788349456588;-0.6914418401393503;-0.222427214243385;0.10085923680346595;-1.2854759002479768;3.0514712823010592;96.27749633789062;0.0;0.019941750242510378;1.206475889671878;0.6511748444573582;-0.7432136917304829;-0.04410194765170564;-0.4881893857445329;3.043801378361632;96.36963653564453;0.0"
        for i in range(30):
            self.publish_from_csv(csv_position)
            rospy.sleep(0.1)
 

    def start_stir(self):
        lst = self.get_right_arm_list()
        current = lst.copy()
        current[6] = 2.8
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01)
        rospy.sleep(0.5)       
        current[6] = 3.04 
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01)     
        rospy.sleep(0.5)     
        current[6] = 2.8
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01)
        rospy.sleep(0.5)        
        current[6] = 3.04 
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01) 
        rospy.sleep(0.5) 
        current[6] = 2.8
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01) 
        rospy.sleep(0.5)       
        current[6] = 3.04 
        for _ in range(15):
            self.set_right_arm_with_tensor(current)
            rospy.sleep(0.01)   

    def return_spoon(self):
        for _ in range(30):
            self.set_right_arm_with_tensor([-0.07133010663667173, 1.8990682154021423, 0.7735098122912198, -0.7205874751091731, -0.06212622190935926, -1.3196069727784272, 3.0430343879676895, 14.657980918884277, 29.462366104125977])
            rospy.sleep(0.01)                 