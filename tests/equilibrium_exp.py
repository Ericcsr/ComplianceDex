import pybullet as pb
import numpy as np
import time
from rigidBodySento import create_primitive_shape, apply_external_world_force_on_local_point, move_object_local_frame

c = pb.connect(pb.GUI)

obj = create_primitive_shape(pb, 0.1, pb.GEOM_BOX, dim=(0.1,0.2,0.03), color = (0.6,0,0,0.7))

forces = np.array([[-2.0,0.0,0.0], 
                   [2.0,0.0,0.0], 
                   [0.0,-2.0,0.0], 
                   [0.0,2.0,0.0]])

poses = np.array([[0.1, 0.0, 0.0],[-0.1, 0.0, 0.0],[0.0, 0.2, 0.01],[0.0, -0.2, -0.01]])
ref = np.array([[0.05, 0.0, 0.0],[-0.05, 0.0, 0.0],[0.0, 0.15, 0.0],[0.0, -0.15, -0.0]])
kp = 2.0 / 0.05
kd = np.sqrt(kp)
vis_sp = []
for i in range(4):
    vis_sp.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=(0.01,), color=(0.3,1,0.3,1), init_xyz=poses[i], collidable=False))
move_object_local_frame(pb, obj, vis_sp, poses)
input()
for s in range(200):
    # Need map force to world frame
    world_coords = np.asarray(move_object_local_frame(pb, obj, vis_sp, poses))
    #forces = (ref - world_coords) * kp
    #print(forces)
    for i in range(4):
        apply_external_world_force_on_local_point(pb, obj, -1, forces[i], poses[i])
    
    pb.stepSimulation()
    time.sleep(0.1)