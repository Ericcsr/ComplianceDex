import time
import pybullet as pb
import numpy as np
from argparse import ArgumentParser

from rigidBodySento import create_primitive_shape

NUM_TIPS = 3

def load_data(exp_name, env_id):
    tips_traj = np.load(f"../data/tips_traj/{exp_name}.npy")[:,env_id]
    obj_traj = np.load(f"../data/obj_traj/{exp_name}.npy")[:,env_id]
    target_traj = np.load(f"../data/target_traj/{exp_name}.npy")[:,env_id]
    compliance_traj = np.load(f"../data/compliance_traj/{exp_name}.npy")[:,env_id]
    goal_traj = np.load(f"../data/goal_traj/{exp_name}.npy")[:,env_id] + 0.15
    
    return tips_traj, obj_traj, target_traj, compliance_traj, goal_traj

def set_tip_pos(oids, positions):
    for i in range(NUM_TIPS):
        pb.resetBasePositionAndOrientation(oids[i], positions[i], [0,0,0,1])

def set_object_pos(oid, pose):
    pb.resetBasePositionAndOrientation(oid, pose[:3], pose[3:7][[1,2,3,0]])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--env_id", type=int, default=0)
    args = parser.parse_args()
    #tips_traj, obj_traj, target_traj, compliance_traj, goal_traj = load_data(args.exp_name, args.env_id)
    c = pb.connect(pb.GUI)

    tips_id = []
    tars_id = []
    tips_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(1,0,0,1)))
    tips_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0,1,0,1)))
    tips_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0,0,1,1)))
    tips_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(1,1,1,1)))
    tars_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0.4,0,0,1)))
    tars_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0,0.4,0,1)))
    tars_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0,0,0.4,1)))
    tars_id.append(create_primitive_shape(pb, 0.01, pb.GEOM_SPHERE, dim=[0.01], color=(0.4,0.4,0.4,1)))
    
    o_id = pb.loadURDF("../../assets/urdf/objects/cube_visualization.urdf")

    g_id = pb.loadURDF("../../assets/urdf/objects/cube_visualization.urdf")


    # tip_pos = np.array([[-0.051, 0.03, -0.04],[0.051,-0.04, 0.03],[0.051,0.0, 0.03],[0.051, 0.04, 0.03]])
    # tip_pos_new = np.array([[-5.0021e-02,  9.9771e-09,  2.8942e-02],
    #                         [ 5.1000e-02, -4.0000e-02,  3.0000e-02],
    #                         [ 5.1000e-02,  0.0000e+00,  3.0000e-02],
    #                         [ 5.1000e-02,  4.0000e-02,  3.0000e-02]])
    # tar_pos = np.array([[-0.03,0., 0.03],[0.03,-0.04, 0.03],[0.03,0., 0.03],[0.03,0.04, 0.03]])
    tip_pos = np.array([[-0.051, 0., 0.03],[0.03,-0.051, 0.03],[0.03, 0.051, 0.03]])
    tar_pos = np.array([[-0.0,0., 0.03],[0.03,-0.03, 0.03],[0.03,0.03, 0.03]])

    set_tip_pos(tars_id, tar_pos)
    set_tip_pos(tips_id, tip_pos)
    while True:
        pass
    # for i in range(len(tips_traj)):
    #     set_tip_pos(tips_id, tips_traj[i])
    #     set_tip_pos(tars_id, target_traj[i])
    #     set_object_pos(o_id, obj_traj[i])
    #     set_object_pos(g_id, goal_traj[i])
    #     time.sleep(0.05)
