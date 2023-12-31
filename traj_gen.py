from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
import pybullet as pb
from argparse import ArgumentParser
import torch
import numpy as np
import time
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
from scipy.spatial.transform import Rotation


parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--grasp_idx", type=int, default=0)
args = parser.parse_args()


tensor_args = TensorDeviceType()
world_file = "my_scene.yml"
robot_file = "iiwa14.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_file,
    tensor_args,
    trajopt_tsteps=40,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=True,
)
motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()

joint_pose = torch.tensor([-0.7153848657390822, 0.23627692865494376, -0.06146527133579401, -1.2628601611175012, 0.01487889923612773, 1.6417360407890011, -2.344269879142319]).cuda()
# #Should be forward kinematics
# state = motion_gen.rollout_fn.compute_kinematics(
#     JointState.from_position(retract_cfg.view(1, -1))
# )
wrist_pose = np.load(f"data/wrist_{args.exp_name}.npy")[args.grasp_idx]
wrist_ori = Rotation.from_euler("XYZ",wrist_pose[3:]).as_quat()[[3,0,1,2]]
target_pose = Pose(torch.from_numpy(wrist_pose[:3]).float().cuda(), quaternion=torch.from_numpy(wrist_ori).float().cuda())
start_state = JointState.from_position(joint_pose.view(1, -1))
t_start = time.time()
result = motion_gen.plan(
        start_state,
        target_pose,
        enable_graph=True,
        enable_opt=True,
        max_attempts=10,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
print("Time taken: ", time.time()-t_start)
print(result.status, result.success, result.attempts)
traj = result.get_interpolated_plan()
print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
print(result.optimized_plan.position.shape)
pb.connect(pb.GUI)

r = pb.loadURDF("curobo_ws/curobo/src/curobo/content/assets/robot/kuka_allegro/model.urdf", useFixedBase=True, flags=pb.URDF_MERGE_FIXED_LINKS)
scene = pb.loadURDF("curobo_ws/curobo/src/curobo/content/assets/scene/nvblox/scene.urdf", useFixedBase=True)
input()
for t in range(len(traj)):
    position = traj.position[t].cpu().numpy()
    print(position)
    for j in range(7):
        pb.resetJointState(r, j, position[j])
    input()
    time.sleep(0.1)

np.save("traj.npy", traj.position.cpu().numpy())