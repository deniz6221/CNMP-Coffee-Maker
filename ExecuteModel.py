import torch
import torch.nn as nn
import numpy as np
import rospy
from CustomBaxter import CustomBaxter
from ultralytics import YOLO
from CustomBaxter import GripperAction

rospy.init_node("ExecuteModel")
baxter = CustomBaxter()

rospy.sleep(1)


d_x , d_y = (5,9)
normalize_tensor_sample = np.array([[1] + [640, 640, 480, 480] + [3.14 for _ in range(d_y -2 )] + [100, 100]])

denormalize_tensor = np.array([[3.14 for _ in range(d_y -2 )] + [100, 100]])
class CNMP(nn.Module):
    def __init__(self):
        super(CNMP, self).__init__()
        
        # Encoder takes observations which are (X,Y) tuples and produces latent representations for each of them
        self.encoder = nn.Sequential(
        nn.Linear(d_x+d_y,128),nn.ReLU(),
        nn.Linear(128,128),nn.ReLU(),
        nn.Linear(128,128)
        )
        
        #Decoder takes the (r_mean, target_t) tuple and produces mean and std values for each dimension of the output
        self.decoder = nn.Sequential(
        nn.Linear(128+d_x,128),nn.ReLU(),
        nn.Linear(128,128),nn.ReLU(),
        nn.Linear(128,2*d_y)
    )

    def forward(self,observations,target_t):
        r = self.encoder(observations) # Generating observations
        r_mean = torch.mean(r,dim=0) # Taking mean and generating the general representation
        r_mean = r_mean.repeat(target_t.shape[0],1) # Duplicating general representation for every target_t
        concat = torch.cat((r_mean,target_t),dim=-1) # Concatenating each target_t with general representation
        output = self.decoder(concat) # Producing mean and std values for each target_t
        return output
    



Pour_Left =  CNMP().double()
Pour_Left.load_state_dict(torch.load("./models/Pour_Left_New.pth"))

Pour_Left.eval()


Pour_Right =  CNMP().double()
Pour_Right.load_state_dict(torch.load("./models/Pour_Right_New.pth"))

Pour_Right.eval()   


Put_Spoon = CNMP().double()
Put_Spoon.load_state_dict(torch.load("./models/Put_Spoon_New.pth"))

Put_Spoon.eval()

image_path = "./snapshot.jpeg"
baxter.save_image(image_path)

yolo_model = YOLO("yolov8x")
results = yolo_model(image_path)
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
    raise

xmin = results_list[middle_cup_idx][0]
xmax = results_list[middle_cup_idx][2]
ymin = results_list[middle_cup_idx][1]
ymax = results_list[middle_cup_idx][3]




first_gamma_val = [xmin, xmax, ymin, ymax]
first_gamma_param = torch.tensor(first_gamma_val, dtype=torch.double)
second_gamma_param = torch.tensor(first_gamma_val, dtype=torch.double)

xvals = np.arange(0,1,0.001)
xvals_tensor = torch.tensor(xvals.reshape(-1, 1), dtype=torch.double)
first_gamma_param = first_gamma_param.expand(xvals_tensor.size(0), -1)
first_gamma_param_norm = first_gamma_param / torch.tensor([[640, 640, 480, 480]], dtype=torch.double)


Pour_Left_Condition = torch.tensor([[0] + first_gamma_val +  baxter.get_left_arm_list()], dtype=torch.double) / normalize_tensor_sample

Pour_Left_Out = Pour_Left(Pour_Left_Condition, torch.cat((xvals_tensor,  first_gamma_param_norm), dim=1))
#Pour_Left_NP = Pour_Left_Out.detach().numpy()


Pour_Right_Condition = torch.tensor([[0] + first_gamma_val +  baxter.get_right_arm_list()], dtype=torch.double) / normalize_tensor_sample

Pour_Right_Out = Pour_Right(Pour_Right_Condition, torch.cat((xvals_tensor,  first_gamma_param_norm), dim=1))
#Pour_Right_NP = Pour_Right_Out.detach().numpy()

baxter.enable_robot_joints(True)
print("Enabling Robot")
rospy.sleep(2)

print("Callibrating Grippers")
baxter.left_gripper_action(GripperAction.CALIBRATE)

baxter.right_gripper_action(GripperAction.CALIBRATE)
print("Callibrated")


for v in Pour_Left_Out:
    denormalized_tensor = v.detach()[0:9] * denormalize_tensor
    baxter.set_left_arm_with_tensor(denormalized_tensor.tolist()[0])
    rospy.sleep(0.01)


for v in Pour_Right_Out:
    denormalized_tensor = v.detach()[0:9] * denormalize_tensor
    baxter.set_right_arm_with_tensor(denormalized_tensor.tolist()[0])
    rospy.sleep(0.01)    


baxter.ask_for_spoon()

rospy.sleep(1) # Wait for spoon

Put_Spoon_Condition = torch.tensor([[0] + first_gamma_val +  baxter.get_right_arm_list()], dtype=torch.double) / normalize_tensor_sample

Put_Spoon_Out = Put_Spoon(Put_Spoon_Condition, torch.cat((xvals_tensor,  first_gamma_param_norm), dim=1))

for v in Put_Spoon_Out:
    denormalized_tensor = v.detach()[0:9] * denormalize_tensor
    baxter.set_right_arm_with_tensor(denormalized_tensor.tolist()[0])
    rospy.sleep(0.01)

baxter.start_stir()

baxter.return_spoon()