import rospy
from CustomBaxter import CustomBaxter
from CustomBaxter import GripperAction
import os
import time
from ultralytics import YOLO


data_dir_name = "Pour_R15"
record_gamma = True
gamma_mean = False
img_path = "gamma_img.png"

if os.path.exists(f"./training_data/{data_dir_name}"):
    print("Directory Already Exists")
    quit(0)

os.makedirs(f"./training_data/{data_dir_name}")

data_dir = f"./training_data/{data_dir_name}" 


data_csv = open(f"{data_dir}/robotdata.csv", "w")



rospy.init_node("data_collector")

baxter = CustomBaxter()



rospy.sleep(2)
baxter.enable_robot_joints(True)

baxter.left_gripper_action(GripperAction.CALIBRATE)
baxter.right_gripper_action(GripperAction.CALIBRATE)


""" rospy.loginfo("Calibrating Grippers")c
baxter.left_gripper.calibrate(True)
baxter.right_gripper.calibrate(True)
rospy.sleep(1)
rospy.loginfo("Calibration Done")
 """

baxter.save_image(f"{data_dir}/{img_path}")


xmax = 0
xmin = 0
ymax = 0
ymin = 0
gamma_lst = []



if (record_gamma):
    yolo_model = YOLO("yolov8x")
    results = yolo_model(f"{data_dir}/{img_path}")
    results_list = results[0].boxes.data.tolist()
    results_list.sort()
    cups_found = 0
    middle_cup_idx = 0
    for k in range(len(results_list)):
        det = results_list[k]
        if (det[5] == 41):
            if (det[2] <= 450 and det[0] >= 190):
                cups_found += 1
                middle_cup_idx = k

    if cups_found != 1:
        print("Middle Cup Not Found or multiple cups on that region.")
        quit(0)
    xmin = results_list[middle_cup_idx][0]
    xmax = results_list[middle_cup_idx][2]
    ymin = results_list[middle_cup_idx][1]
    ymax = results_list[middle_cup_idx][3]


if (not gamma_mean):
    gamma_lst.append(xmin)
    gamma_lst.append(xmax)
    gamma_lst.append(ymin)
    gamma_lst.append(ymax)
else:
    gamma_lst.append((xmin + xmax)/ 2)
    gamma_lst.append((ymin + ymax) / 2)

gamma_str = ""
for gm in gamma_lst:
    gamma_str += f"{gm};"

gamma_str = gamma_str[:-1]


baxter.go_to_final_pos_right()
print("Ready For Collection")
while rospy.is_shutdown() is False:

    if (baxter.record_button_state or baxter.right_record_button_state):
        NOW = int(time.time_ns() / 1_000_000)
        if (not record_gamma):
            line = f"{NOW};{baxter.get_right_arm_training_data()}\n"
        else:
            line = f"{NOW};{gamma_str};{baxter.get_right_arm_training_data()}\n"
        data_csv.write(line)
        rospy.sleep(0.001) 



