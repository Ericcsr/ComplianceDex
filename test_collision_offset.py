import pybullet as pb
import numpy as np
import torch
from rigidBodySento import create_primitive_shape
from differentiable_robot_model import DifferentiableRobotModel

pb.connect(pb.GUI)

joint_angles = [np.pi/6, -np.pi/9, np.pi/6, np.pi/6,
                 np.pi/6, 0.0     , np.pi/6, np.pi/6,
                 np.pi/6, np.pi/9 , np.pi/6, np.pi/6,
                 np.pi/3, np.pi/4 , np.pi/6, np.pi/6]

urdf_link = "pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"
r = pb.loadURDF(urdf_link, useFixedBase=True)

for i, angle in enumerate(joint_angles):
    pb.resetJointState(r, i, angle)

robot_model = DifferentiableRobotModel(urdf_link, device="cpu")

#collisions_links = ["dip","dip_2","dip_3", "thumb_dip"]
#collisions_offsets = [[0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, 0.02, -0.015]]
#collisions_links = ["pip", "pip_2", "pip_3"]
#collisions_offsets = [[0.01, 0.0, -0.01]] * 3
collisions_links = ["mcp_joint", "mcp_joint_2", "mcp_joint_3"]
collisions_offsets = [[-0.02, 0.04, 0.015]] * 3

poses = robot_model.compute_forward_kinematics(torch.tensor(joint_angles), collisions_links, offsets=collisions_offsets, recursive=True)[0].view(-1, 3)

color_code = [[1,0,0,0.7],[0,1,0,0.7],[0,0,1,0.7],[0,1,1,0.7]]

for i, (link, offset) in enumerate(zip(collisions_links, collisions_offsets)):
    o = create_primitive_shape(pb, 0.0, shape=pb.GEOM_SPHERE, dim=(0.02,), collidable=False, color=color_code[i])
    pb.resetBasePositionAndOrientation(o, poses[i], [0,0,0,1])


input("Press Enter to end")


