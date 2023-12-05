from argparse import ArgumentParser
import json
import open3d as o3d
import numpy as np
import torch
import optimize_pregrasp as opt
import verify_pregrasp as sim
from gpis import GPIS
import pybullet as pb

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--num_iters", type=int, default=1000)
parser.add_argument("--mode", type=str, default="prob")
parser.add_argument("--use_config", action="store_true", default=False)
parser.add_argument("--visualize", action="store_true", default=False)
parser.add_argument("--wrist_x", type=float, default=0.0)
parser.add_argument("--wrist_y", type=float, default=0.0)
parser.add_argument("--wrist_z", type=float, default=0.0)
parser.add_argument("--num_repeat", type=int, default=5) 
args = parser.parse_args()

if args.use_config:
    config = json.load(open(f"assets/{args.exp_name}/config.json"))
    args.wrist_x = config["wrist_x"]
    args.wrist_y = config["wrist_y"]
    args.wrist_z = config["wrist_z"]
    args.floor_offset = config["floor_offset"]

mesh = o3d.io.read_triangle_mesh(f"assets/{args.exp_name}/{args.exp_name}.obj")
pcd = mesh.sample_points_poisson_disk(4096)
friction = 0.5
gpis = GPIS(0.08, 1) # 0.02, 0.1
gpis.load_state_data(f"{args.exp_name}_state")
robot_urdf = "pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"

# simulation validator
c = pb.connect(pb.GUI if args.visualize else pb.DIRECT)
pb.setTimeStep(0.001)
pb.setGravity(0.0, 0.0, -1.0)
validator = sim.LeapHandValidator(pb, object_urdf=sim.object_dict[args.exp_name], 
                                  init_object_pose=[0,0,0,0,0,0,1],init_robot_pos=[args.wrist_x,args.wrist_y,args.wrist_z], uid=c, floor_offset=args.floor_offset, friction=friction)

optimizers = {
    "kingpis": opt.KinGPISGraspOptimizer,
    "prob": opt.ProbabilisticGraspOptimizer,
    "wc": opt.WCKinGPISGraspOptimizer,
    "sdf": opt.SDFGraspOptimizer,
    "gpis": opt.GPISGraspOptimizer
}

if args.mode in ["kingpis", "wc", "prob"]:
    grasp_optimizer = optimizers[args.mode](robot_urdf,
                                            tip_bounding_box=[opt.FINGERTIP_LB, opt.FINGERTIP_UB],
                                            optimize_target=True,
                                            num_iters=args.num_iters,
                                            palm_offset=[opt.WRIST_OFFSET[0]+args.wrist_x,opt.WRIST_OFFSET[1]+args.wrist_y,opt.WRIST_OFFSET[2]+args.wrist_z])
elif args.mode in ["sdf", "gpis"]:
    grasp_optimizer = optimizers[args.mode](tip_bounding_box=[opt.FINGERTIP_LB, opt.FINGERTIP_UB],
                                            optimize_target=True, num_iters=args.num_iters)


def single_experiment():
    validator.reset() # TODO: Implement reset
    init_tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).double().cuda()
    init_joint_angle = torch.tensor([[np.pi/12, -np.pi/9, np.pi/8, np.pi/8,
                                    np.pi/12, 0.0     , np.pi/8, np.pi/8,
                                    np.pi/12, np.pi/9 , np.pi/8, np.pi/8,
                                    np.pi/2.5, np.pi/3 , np.pi/6, np.pi/6]]).double().cuda() + torch.randn(1,16).double().cuda() * 0.1

    # initialize target locatino
    rand_n = torch.rand(4,1)
    target_pose = rand_n * torch.tensor(opt.FINGERTIP_LB).view(-1,3) + (1 - rand_n) * torch.tensor(opt.FINGERTIP_UB).view(-1,3).double()
    target_pose = 0.2 * target_pose.unsqueeze(0).cuda()

    if args.mode == "wc":
        compliance = torch.tensor([[200.0,200.0, 200.0, 200.0]]).double().cuda()
    else:
        compliance = torch.tensor([[10.0, 10.0, 10.0, 20.0]]).double().cuda()

    if args.mode in ["sdf", "gpis"]:
        opt_tip_pose, opt_compliance, opt_target_pose, success_flag = grasp_optimizer.optimize(init_tip_pose, target_pose, compliance, friction, mesh if args.mode == "sdf" else gpis, verbose=False)
    else:
        joint_angles, opt_compliance, opt_target_pose, success_flag = grasp_optimizer.optimize(init_joint_angle,target_pose, compliance, friction, gpis, verbose=False)
        opt_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1, 3)
    if success_flag == False:
        return False
    
    if args.visualize:
        tips, targets = opt.vis_grasp(opt_tip_pose, opt_target_pose)
        o3d.visualization.draw_geometries([pcd, *tips, *targets])
    # simulate for verification
    finger_pose = opt_tip_pose.detach().squeeze().cpu().numpy()
    target_pose = opt_target_pose.detach().squeeze().cpu().numpy()
    center = target_pose.sum(axis=0) / 4
    finger_pose = finger_pose - center
    kp = opt_compliance.detach().squeeze().cpu().numpy().repeat(3).reshape(-1,3)
    kd = np.sqrt(kp) * 0.8
    #print(finger_pose, target_pose, kp, kd)
    obj_pos, obj_orn = validator.execute_grasp(finger_pose, target_pose, [0,0,0,0,0,0,1], kp, kd, visualize=args.visualize, pause=False)
    if obj_pos[2] < -0.2: # Dropped too low.
        return False
    return True

success_count = 0
for i in range(args.num_repeat):
    success_count += int(single_experiment())
    print(f"Experiment {i}:",success_count)

print(f"Success rate: {success_count / args.num_repeat}")


    
