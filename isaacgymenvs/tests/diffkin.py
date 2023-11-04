from differentiable_robot_model import DifferentiableRobotModel
#import pytorch_kinematics as pk
import torch
import pybullet as pb

OFFSETS = [[0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.05, -0.015]]

c = pb.connect(pb.DIRECT)

urdf_path = "../grasping/pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"
robot = DifferentiableRobotModel(urdf_path)
r = pb.loadURDF(urdf_path, useFixedBase=True)

joint_pos = torch.zeros(16, dtype=torch.float32).requires_grad_(True)
joint_vel = torch.zeros(16, dtype=torch.float32).requires_grad_(True)

ee_pos, ee_quat = robot.compute_forward_kinematics(joint_pos, ["fingertip", "fingertip_2", "fingertip_3", "thumb_fingertip"],offsets=OFFSETS)


link_state = pb.getLinkState(r, 3, computeForwardKinematics=True)
ee_pos_, ee_quat_ = link_state[0], link_state[1]

print(ee_pos, ee_pos_)
print(ee_quat, ee_quat_)
