from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
import pybullet as pb
import torch
import time
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    )
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

tensor_args = TensorDeviceType()
world_file = "collision_mesh_scene.yml"
robot_file = "franka.yml"
motion_gen_config = MotionGenConfig.load_from_robot_config(
    robot_file,
    world_file,
    tensor_args,
    trajopt_tsteps=40,
    collision_checker_type=CollisionCheckerType.MESH,
    use_cuda_graph=False,
)
motion_gen = MotionGen(motion_gen_config)
robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
retract_cfg = motion_gen.get_retract_config()
state = motion_gen.rollout_fn.compute_kinematics(
    JointState.from_position(retract_cfg.view(1, -1))
)

retract_pose = Pose(state.ee_pos_seq.squeeze()+torch.tensor([0.3,0.0,-0.2]).cuda(), quaternion=state.ee_quat_seq.squeeze())
start_state = JointState.from_position(retract_cfg.view(1, -1))
t_start = time.time()
result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=False,
        enable_opt=True,
        max_attempts=1,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
print("Time taken: ", time.time()-t_start)
t_start = time.time()
result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=False,
        enable_opt=True,
        max_attempts=1,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
print("Time taken: ", time.time()-t_start)
t_start = time.time()
result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=False,
        enable_opt=True,
        max_attempts=1,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
print("Time taken: ", time.time()-t_start)
t_start = time.time()
result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=False,
        enable_opt=True,
        max_attempts=1,
        num_trajopt_seeds=10,
        num_graph_seeds=10)
print("Time taken: ", time.time()-t_start)
print(result.status, result.success, result.attempts)
traj = result.get_interpolated_plan()
print("Trajectory Generated: ", result.success, result.optimized_dt.item(), traj.position.shape)
input()
pb.connect(pb.GUI)

r = pb.loadURDF("curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf", useFixedBase=True, flags=pb.URDF_MERGE_FIXED_LINKS)
scene = pb.loadURDF("curobo/src/curobo/content/assets/scene/nvblox/scene.urdf", useFixedBase=True, basePosition=[1.3, 0.080, 1.7], baseOrientation = [-0.471, 0.284, 0.834,0.043])

for t in range(len(traj)):
    position = traj.position[t].cpu().numpy()
    for j in range(7):
        pb.resetJointState(r, j, position[j])
    input()
    time.sleep(0.1)
