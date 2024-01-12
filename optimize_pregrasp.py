import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from torchsdf import compute_sdf
from gpis import GPIS
import torch
import copy
from functools import partial
from differentiable_robot_model.robot_model import DifferentiableRobotModel
from pybullet_robot.robots import robot_configs
from math_utils import minimum_wrench_reward, euler_angles_to_matrix
from pb_grasp_visualizer import GraspVisualizer


EE_OFFSETS = [[0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.05, -0.015]]

# Initial guess for wrist pose
WRIST_OFFSET = np.array([[-0.05, 0.0, 0.06, 0.0, 0.0, 0.0],#])
                         [-0.04, 0.03, 0.03, 0.0,0.0,-np.pi/4],
                         [-0.04, -0.03, 0.03, 0.0,0.0,np.pi/4],
                         [0.1, 0.06, -0.05, -np.pi/2,np.pi/2,0.0],
                         [-0.0, -0.06, 0.03, 0.0,0.0,np.pi/2],
                         [0.02, -0.04, 0.03, 0.0,0.0,3 * np.pi/4]]) # [0.02]
#WRIST_OFFSET = [-0.58, -0.05, -0.19]

z_margin = 0.3
#FINGERTIP_LB = [-0.01, -0.1, -z_margin, -0.01, -0.03, -z_margin, -0.01, 0.03, -z_margin, -0.1, -0.05, -z_margin]
FINGERTIP_LB = [-0.2, -0.2,   0.015,   -0.2, -0.2,      0.015,  -0.2, -0.2,      0.015, -0.2, -0.2, 0.015]
FINGERTIP_UB = [0.2,  0.2,  z_margin,  0.2,   0.2,   z_margin,   0.2,  0.2,  z_margin,   0.2,  0.2,  z_margin]

def vis_grasp(tip_pose, target_pose):
    tip_pose = tip_pose.cpu().detach().numpy().squeeze()
    target_pose = target_pose.cpu().detach().numpy().squeeze()
    tips = []
    targets = []
    color_code = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0]])
    for i in range(4):
        tip = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        tip.paint_uniform_color(color_code[i])
        tip.translate(tip_pose[i])
        target = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        target.paint_uniform_color(color_code[i] * 0.5)
        target.translate(target_pose[i])
        tips.append(tip)
        targets.append(target)
    return tips, targets

#@torch.jit.script
def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    weight = torch.nn.functional.normalize(weight, dim=1, p=1.0)
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = S1.mean(dim=1, keepdim=True) # [num_envs, 3]
    c2 = S2.mean(dim=1, keepdim=True)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H+1e-6*torch.rand_like(H).cuda())
    V = Vh.mH
    R_ = V @ U.transpose(1,2)
    mask = R_.det() < 0.0
    sign = torch.ones(R_.shape[0], 3, 3).cuda()
    sign[mask,:, -1] *= -1.0
    R = (V*sign) @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

# Assume friction is uniform
# Differentiable 
def force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal, mass=0.4, gravity=None, M=0.2, COM=[0.0, 0.05, 0.0]):
    """
    Params:
    tip_pose: world frame [num_envs, num_fingers, 3]
    target_pose: world frame [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    friction_mu: scaler
    current_normal: world frame [num_envs, num_fingers, 3]
    
    Returns:
    reward: [num_envs]
    """
    # Prepare dummy gravity
    # Assume COM is at center of target
    if COM is not None:
        com = torch.tensor(COM).cuda().double()
    if gravity is not None:
        dummy_tip = torch.zeros(tip_pose.shape[0], 1, 3).cuda()
        dummy_tip[:,0,:] = target_pose.mean(dim=1) if COM is None else com
        dummy_target = torch.zeros(target_pose.shape[0], 1, 3).cuda()
        dummy_target[:,0,2] = -M # much lower than center of mass
        dummy_compliance = gravity * mass/M * torch.ones(compliance.shape[0], 1).cuda()
        R,t = optimal_transformation_batch(torch.cat([tip_pose, dummy_tip], dim=1), 
                                           torch.cat([target_pose, dummy_target], dim=1), 
                                           torch.cat([compliance, dummy_compliance], dim=1))
    else:
        R,t = optimal_transformation_batch(tip_pose, target_pose, compliance)
    # tip position at equilirium
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose
    force = compliance.unsqueeze(2) * (-diff_vec)
    dir_vec = diff_vec / diff_vec.norm(dim=2).unsqueeze(2)
    # Rotate local norm to equilibrium pose
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos similarity between force direction and surface normal
    # dir_vec: [num_envs, num_fingers, 3]
    # normal_eq: [num_envs, num_fingers, 3]
    ang_diff =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq) # Here the bug
    cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
    margin = (ang_diff - cos_mu).clamp(min=-0.9999)
    # if (margin == -0.999).any():
    #     print("Debug:",dir_vec, normal_eq)
    # we hope margin to be as large as possible, never below zero
    force_norm = force.norm(dim=2)
    reward = (0.2 * torch.log(ang_diff+1)+ 0.8 * torch.log(margin+1)).sum(dim=1)
    return reward , margin, force_norm, R, t
    
def check_force_closure(tip_pose, target_pose, compliance, R, t, COM, mass, gravity):
    #R, t = optimal_transformation_batch(tip_pose.cuda().unsqueeze(0), target_pose.cuda().unsqueeze(0), compliance.cuda().unsqueeze(0))
    tip_pose = tip_pose.cpu().detach()
    target_pose = target_pose.cpu().detach()
    compliance = compliance.cpu().detach()
    R = R.squeeze().cpu().detach()
    t = t.squeeze().cpu().detach()
    
    tip_pose_after = (R@tip_pose.transpose(0,1)).transpose(0,1) + t
    force = compliance.unsqueeze(1) * (target_pose - tip_pose_after)
    gravity_force = (torch.tensor([0.0, 0.0, -0.2])- R@COM - t) * gravity * mass/0.2
    COM_after = (R@COM + t).cpu().detach()
    total_torque = ((tip_pose_after - COM_after).cross(force)).sum(dim=0)
    # total_torque += gravity_force.cross(torch.from_numpy(COM))
    total_force = force.sum(dim=0) + gravity_force
    print(total_torque, total_force)


class KinGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=[-0.01, 0.015, 0.12],
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 mass=0.1, com = [0.0,0.0,0.0],
                 gravity=True,
                 uncertainty=0.0):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).cuda()
        self.optimize_target = optimize_target
        self.gravity = gravity
        self.mass = mass
        self.com = com

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names, 
                                                           offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)

    def optimize(self, joint_angles, target_pose, compliance, friction_mu, object_mesh, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        triangles = np.asarray(object_mesh.triangles)
        vertices = np.asarray(object_mesh.vertices)
        face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
        object_mesh.scale(0.9, center=[0,0,0])
        vertices = np.asarray(object_mesh.vertices)
        face_vertices_deflate = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.Adam([{"params":joint_angles, "lr":2e-3},
                                        {"params":target_pose, "lr":1e-5},
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.Adam([{"params":joint_angles, "lr":1e-2}, # Directly optimizing joint angles can result in highly non-linear manifold..
                                        {"params":compliance, "lr":0.2}])
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(joint_angles.shape[0]).cuda()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.forward_kinematics(joint_angles)
            _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
            dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
            tar_dist, tar_sign, _, _ = compute_sdf(target_pose.view(-1,3), face_vertices)
            # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
            current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
            current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
            task_reward, margin, force_norm = force_eq_reward(all_tip_pose.view(target_pose.shape), 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(target_pose.shape),
                                mass = self.mass, COM=self.com,
                                gravity=10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.sqrt(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * (tar_sign * torch.sqrt(tar_dist).view(target_pose.shape[0], target_pose.shape[1])).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), compliance)
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_joint_angle, opt_compliance, opt_target_pose, flag

class SDFGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False, mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=0.0):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target
        self.mass = mass
        self.com = com
        self.gravity = gravity

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, object_mesh, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        TODO: Add a penalty term to encourage target pose stay inside the object.
        Params:
        tip_pose: [num_envs, num_fingers, 3]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        tip_pose = tip_pose.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        triangles = np.asarray(object_mesh.triangles)
        vertices = np.asarray(object_mesh.vertices)
        face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
        object_mesh.scale(0.9, center=[0,0,0])
        vertices = np.asarray(object_mesh.vertices)
        face_vertices_deflate = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":target_pose, "lr":1e-3}, 
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        opt_tip_pose = tip_pose.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cuda()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = tip_pose.view(-1,3)
            _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
            dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
            tar_dist, tar_sign, _, _ = compute_sdf(target_pose.view(-1,3), face_vertices)
            # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
            current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
            current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
            task_reward, margin, force_norm = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.sqrt(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *(torch.sqrt(tar_dist).view(tip_pose.shape[0], tip_pose.shape[1]) * tar_sign).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()))
            if torch.isnan(l.sum()):
                print(dist, tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_tip_pose, opt_compliance, opt_target_pose, flag

class GPISGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False, mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=20.0):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.uncertainty = uncertainty

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        TODO: Add a penalty term to encourage target pose stay inside the object.
        Params:
        tip_pose: [num_envs, num_fingers, 3]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        """
        tip_pose = tip_pose.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":target_pose, "lr":1e-3}, 
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        opt_tip_pose = tip_pose.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).double().cuda()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = tip_pose.view(-1,3)
            dist, var = gpis.pred(all_tip_pose)
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose)
            task_reward, margin, force_norm = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 25.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.abs(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *tar_dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            variance_cost = self.uncertainty * var.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + variance_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()), float(variance_cost.sum()))
            if torch.isnan(l.sum()):
                print("Loss NaN trace:",dist, tip_pose, target_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                compliance.clamp_(min=40.0) # prevent negative compliance
                #target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_tip_pose, opt_compliance, opt_target_pose, flag

class KinGPISGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                 mass=0.1, com=[0.0, 0.0, 0.0], gravity=True, uncertainty=10.0):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().cuda()
        self.optimize_target = optimize_target
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.uncertainty = uncertainty

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names,
                                                                offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)
    
    def optimize(self, joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)

        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":joint_angles, "lr":2e-3},
                                        {"params":target_pose, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":joint_angles, "lr":1e-2}, # Directly optimizing joint angles can result in highly non-linear manifold..
                                        {"params":compliance, "lr":0.2}])
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(joint_angles.shape[0]).double().cuda()
        normal = None
        opt_margin = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.forward_kinematics(joint_angles)
            dist, var = gpis.pred(all_tip_pose)
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose)
            task_reward, margin, force_norm = force_eq_reward(all_tip_pose.view(target_pose.shape), 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(target_pose.shape),
                                mass = self.mass,
                                COM = self.com,
                                gravity = 10.0 if self.gravity else None)
            c = -task_reward * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
            variance_cost = self.uncertainty * torch.log(100 * var).view(target_pose.shape[0], target_pose.shape[1])
            dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * tar_dist.view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), variance_cost.detach())
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_margin = margin
                    opt_value[update_flag] = l[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                compliance.clamp_(min=40.0) # prevent negative compliance
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        if verbose:
            print(opt_margin, normal)
        flag = (opt_margin > 0.0).all()
        return opt_joint_angle, opt_compliance, opt_target_pose, flag
    
class WCKinGPISGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, 
                 optimize_target=False, # dummy
                 ref_q=None,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                 min_force = 5.0,
                 anchor_link_names = None,
                 anchor_link_offsets = None,
                 collision_pairs = None,
                 collision_pair_threshold = 0.02,
                 mass=0.1, com=[0.0, 0.0, 0.0], gravity=True,
                 uncertainty=0.0,
                 optimize_palm=False):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().cuda()
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.min_force = min_force
        self.mass = mass
        self.com = com
        self.uncertainty = uncertainty
        self.gravity = gravity
        self.optimize_palm = optimize_palm
        self.anchor_link_names = anchor_link_names
        self.anchor_link_offsets = anchor_link_offsets
        collision_pairs = torch.tensor(collision_pairs).long().cuda()
        self.collision_pair_left = collision_pairs[:,0]
        self.collision_pair_right = collision_pairs[:,1]
        self.collision_pair_threshold = collision_pair_threshold

    def forward_kinematics(self, joint_angles, palm_poses=None):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :params: palm_offset: [num_envs, 3]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        if palm_poses is None:
            palm_poses = self.palm_offset
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles.float(), self.ee_link_names,
                                                                offsets=self.ee_link_offsets, recursive=False)[0].double().view(-1,4,3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        tip_poses = torch.bmm(R, tip_poses.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        return tip_poses
    
    def compute_collision_loss(self, joint_angles, palm_poses=None):
        if palm_poses is None:
            palm_poses = self.palm_offset
        anchor_pose = self.robot_model.compute_forward_kinematics(joint_angles.float(), 
                                                                  self.anchor_link_names, 
                                                                  offsets=self.anchor_link_offsets, 
                                                                  recursive=False)[0].double().view(-1,len(self.anchor_link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        anchor_pose = torch.bmm(R, anchor_pose.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        collision_pair_left = anchor_pose[:,self.collision_pair_left]
        collision_pair_right = anchor_pose[:,self.collision_pair_right]
        dist = torch.norm(collision_pair_left - collision_pair_right, dim=2) # [num_collision_pairs]
        # Add pairwise collision cost
        mask = dist < self.collision_pair_threshold
        inverse_dist = 1.0 / dist
        inverse_dist[~mask] *= 0.0
        cost =  inverse_dist.sum(dim=1)
        # Add ground collision cost
        z_mask = anchor_pose[:,:,2] < 0.02
        z_dist_cost = 1/(anchor_pose[:,:,2]) * 0.1
        z_dist_cost[~z_mask] *= 0.0
        z_cost = z_dist_cost.sum(dim=1)
        cost += z_cost
        # add palm-floor collision cost
        if self.optimize_palm:
            palm_z_mask = palm_poses[:,2] < 0.02
            palm_z_dist_cost = 1/(palm_poses[:,2])
            palm_z_dist_cost[~palm_z_mask] *= 0.0
            palm_z_cost = palm_z_dist_cost
            cost += palm_z_cost
        return cost

    def optimize(self, init_joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = init_joint_angles.clone().requires_grad_(True)
        #compliance = compliance.clone().requires_grad_(True)
        params_list = [{"params":joint_angles, "lr":1e-3}]
        palm_poses = self.palm_offset[:,:3].clone().requires_grad_(self.optimize_palm)
        palm_oris = self.palm_offset[:,3:].clone().requires_grad_(self.optimize_palm)
        if self.optimize_palm:
            params_list.append({"params":palm_poses, "lr":1e-3})
            params_list.append({"params":palm_oris, "lr":1e-3})

        optim = torch.optim.RMSprop(params_list)
        num_envs = init_joint_angles.shape[0]
        
        opt_joint_angle = joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_palm_poses = self.palm_offset.clone()
        opt_value = torch.inf * torch.ones(num_envs).double().cuda()
        opt_margin = torch.zeros(num_envs, 4).double().cuda()
        opt_R, opt_t = torch.zeros(num_envs, 3, 3).double().cuda(), torch.zeros(num_envs, 3).double().cuda()
        for s in range(self.num_iters):
            optim.zero_grad()
            palm_posori = torch.hstack([palm_poses, palm_oris])
            all_tip_pose = self.forward_kinematics(joint_angles, palm_posori) # [num_envs, num_fingers, 3]

            # Target pose are passively determined.
            dist, var = gpis.pred(all_tip_pose)
            current_normal = gpis.compute_normal(all_tip_pose)
            task_cost = 0.0
            margin = []
            forces = []
            # Using wrench closure formulation, we cannot align multiple contact points with same pregrasp.
            for i in range(num_envs):
                task_cost_, margin_, forces_ = minimum_wrench_reward(all_tip_pose[i], 
                                                                     current_normal[i], 
                                                                     friction_mu, min_force=self.min_force)
                # Compute target positions
                target_pose[i] = all_tip_pose[i].detach() + forces_.detach() / compliance[i].unsqueeze(1)
                task_cost += task_cost_
                margin.append(margin_)
                forces.append(forces_)
            margin = torch.stack(margin)
            forces = torch.stack(forces)
            force_norm = forces.norm(dim=2).view(target_pose.shape[0], target_pose.shape[1])
            forces = forces.view(target_pose.shape[0], target_pose.shape[1], 3)
            
            c = task_cost * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1) # Become a regularization term
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
            variance_cost = self.uncertainty * torch.log(100 * var).view(target_pose.shape[0], target_pose.shape[1])
            dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
            # Add kinematic constraint loss
            if self.optimize_palm:
                palm_dist,_ = gpis.pred(palm_poses)
                palm_dist = palm_dist
                l += 1/palm_dist # Need to ensure palm is outside the object.
                #print("palm dist:", float(palm_dist.mean()))
            l += self.compute_collision_loss(joint_angles, palm_posori)

            l.sum().backward()
            if verbose:
                print("Loss:",float(l.sum()), float(task_cost))
            if torch.isnan(l.sum()):
                print(dist, all_tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum() and s > 20:
                    #opt_margin = margin[update_flag].clone()
                    opt_value[update_flag] = l[update_flag].clone()
                    opt_joint_angle[update_flag] = joint_angles[update_flag].clone()
                    opt_target_pose[update_flag] = target_pose[update_flag].clone()
                    opt_compliance[update_flag] = compliance[update_flag].clone() # dummy
                    opt_palm_poses[update_flag] = torch.hstack([palm_poses, palm_oris])[update_flag].clone()
            optim.step()
        if verbose:
            print(opt_margin)
        return opt_joint_angle, opt_compliance, opt_target_pose, opt_palm_poses, opt_margin, opt_R, opt_t

class ProbabilisticGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names,
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET, # Should be a [num_envs, 6] matrix, 
                 num_iters=1000, optimize_target=False,
                 ref_q=None,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                #  pregrasp_coefficients = [[0.75,0.75,0.75,0.75],
                #                           [0.70,0.70,0.70,0.70],
                #                           [0.65,0.65,0.65,0.65]],
                #  pregrasp_weights = [0.2,0.6,0.2],
                 pregrasp_coefficients = [[0.7,0.7,0.7,0.7]],
                 pregrasp_weights = [1.0],
                 anchor_link_names = None,
                 anchor_link_offsets = None,
                 collision_pairs = None,
                 collision_pair_threshold = 0.02,
                 mass = 0.1, com = [0.0, 0.0, 0.0], gravity=True,
                 uncertainty=20.0,
                 optimize_palm = False,
                 num_samples = 10):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        print("Wrist offset:", palm_offset)
        self.palm_offset = torch.from_numpy(palm_offset).double().cuda()
        self.optimize_target = optimize_target
        self.optimize_palm = optimize_palm
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.pregrasp_coefficients = torch.tensor(pregrasp_coefficients).cuda()
        self.pregrasp_weights = torch.tensor(pregrasp_weights).double().cuda()
        self.anchor_link_names = anchor_link_names
        self.anchor_link_offsets = anchor_link_offsets
        collision_pairs = torch.tensor(collision_pairs).long().cuda()
        self.collision_pair_left = collision_pairs[:,0]
        self.collision_pair_right = collision_pairs[:,1]
        self.collision_pair_threshold = collision_pair_threshold
        self.mass = mass
        self.com = com
        self.gravity = gravity
        self.uncertainty = uncertainty
        self.num_samples = num_samples

    def forward_kinematics(self, joint_angles, palm_poses=None, link_names=None):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :params: palm_offset: [num_envs, 3]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        if palm_poses is None:
            palm_poses = self.palm_offset
        if link_names is None:
            link_names = self.ee_link_names
            offsets = self.ee_link_offsets
        else:
            offsets = [[0.0, 0.0, 0.0]] * len(link_names)
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles.float(), link_names,
                                                                offsets=offsets, recursive=False)[0].double().view(-1,len(link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        tip_poses = torch.bmm(R, tip_poses.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        return tip_poses
    
    def compute_collision_loss(self, joint_angles, palm_poses=None):
        if palm_poses is None:
            palm_poses = self.palm_offset
        anchor_pose = self.robot_model.compute_forward_kinematics(joint_angles.float(), 
                                                                  self.anchor_link_names, 
                                                                  offsets=self.anchor_link_offsets, 
                                                                  recursive=False)[0].double().view(-1,len(self.anchor_link_names),3)
        R = euler_angles_to_matrix(palm_poses[:,3:], convention="XYZ") #[num_envs, 3, 3]
        anchor_pose = torch.bmm(R, anchor_pose.transpose(1,2)).transpose(1,2) + palm_poses[:,:3].unsqueeze(1)
        collision_pair_left = anchor_pose[:,self.collision_pair_left]
        collision_pair_right = anchor_pose[:,self.collision_pair_right]
        dist = torch.norm(collision_pair_left - collision_pair_right, dim=2) # [num_collision_pairs]
        # Add pairwise collision cost
        mask = dist < self.collision_pair_threshold
        inverse_dist = 1.0 / dist
        inverse_dist[~mask] *= 0.0
        cost =  inverse_dist.sum(dim=1)
        # Add ground collision cost
        #z_mask = anchor_pose[:,:,2] < 0.02
        z_dist_cost = 1/(anchor_pose[:,:,2]) * 0.1
        #z_dist_cost[~z_mask] *= 0.0
        z_cost = z_dist_cost.sum(dim=1)
        cost += z_cost
        # add palm-floor collision cost
        if self.optimize_palm:
            palm_z_mask = palm_poses[:,2] < 0.02
            palm_z_dist_cost = 1/(palm_poses[:,2])
            palm_z_dist_cost[~palm_z_mask] *= 0.0
            palm_z_cost = palm_z_dist_cost
            cost += palm_z_cost
        return cost
    
    def compute_contact_margin(self, tip_pose, target_pose, current_normal, friction_mu):
        force_dir = tip_pose - target_pose
        force_dir = force_dir / force_dir.norm(dim=2, keepdim=True)
        ang_diff = torch.einsum("ijk,ijk->ij",force_dir, current_normal)
        cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
        margin = (ang_diff - cos_mu).clamp(-0.999)
        reward = (0.2 * torch.log(ang_diff+1)+ 0.8*torch.log(margin+1)).sum(dim=1)
        return reward

    # assume all_tip_pose has same shape as target_pose
    def compute_loss(self, all_tip_pose, joint_angles, target_pose, compliance, friction_mu, gpis):
        # All tip pose should be [num_envs*1, 4, 3]
        # Should use pregrasp tip_pose for sampling
        dist, var = gpis.pred(all_tip_pose)
        tar_dist, _ = gpis.pred(target_pose)
        current_normal = gpis.compute_normal(all_tip_pose)
        task_reward, margin, force_norm, R, t = force_eq_reward(
                            all_tip_pose,
                            target_pose,
                            compliance,
                            friction_mu, 
                            current_normal.view(target_pose.shape),
                            mass=self.mass,
                            COM = self.com,
                            gravity=10.0 if self.gravity else None)

        # initial feasibility should be equally important as task reward.
        c = -task_reward * 100.0
        center_cost = -self.compute_contact_margin(all_tip_pose, target_pose, current_normal, friction_mu=friction_mu) * 50.0
        offsets = torch.tensor([0.0, 0.0, 0.02]).cuda()
        reg_cost = 10 * (torch.bmm(R,all_tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1) - all_tip_pose - offsets).norm(dim=2).sum(dim=1)
        force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=10.0).sum(dim=1) * 10
        #force_cost = - (all_tip_pose - target_pose).norm(dim=2).sum(dim=1)
        #force_cost = -force_norm.clamp(max=10.0).mean(dim=1) * 20.0
        ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 10.0 # 20.0
        contact_prob = 1.0/(torch.sqrt(var))*torch.exp(-dist**2/(2*var))
        #variance_cost = self.uncertainty * torch.log(100 * var).max(dim=1)[0]
        #print(float(variance_cost.max(dim=1)[0]))
        dist_cost = 5000 * torch.abs(dist).sum(dim=1)
        tar_dist_cost = 10 * tar_dist.sum(dim=1) # 30
        l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost + reg_cost * 200 # Encourage target pose to stay inside the object
        #print("All costs:", float(c.mean()), float(dist_cost.mean()), float(tar_dist_cost.mean()), float(center_cost.mean()), float(force_cost.mean()), float(ref_cost.mean()), float(variance_cost.mean()), float(reg_cost.mean()))
        return l, margin, R, t, contact_prob

    def closure(self, joint_angles, compliance, target_pose, palm_poses, palm_oris, friction_mu, gpis, num_envs):
        self.optim.zero_grad()
        palm_posori = torch.hstack([palm_poses, palm_oris])
        self.pregrasp_tip_pose = self.forward_kinematics(joint_angles, palm_posori)
    
        
        # Compute expected loss
        target_pose_extended = target_pose.repeat(len(self.pregrasp_coefficients),1,1)
        pregrasp_tip_pose_extended = self.pregrasp_tip_pose.repeat(len(self.pregrasp_coefficients),1,1) #[e1,e2,e3,e4,e1,e2,e3,e4, ...]
        pregrasp_coeffs = self.pregrasp_coefficients.repeat_interleave(num_envs,dim=0)
        all_tip_pose = target_pose_extended + pregrasp_coeffs.view(-1, 4, 1) * (pregrasp_tip_pose_extended - target_pose_extended)
        l, margin, R, t, contact_prob = self.compute_loss(all_tip_pose, # [num_envs*num_coeffs, 4, 3]
                                      joint_angles.repeat(len(self.pregrasp_coefficients), 1), 
                                      target_pose_extended, 
                                      compliance.repeat(len(self.pregrasp_coefficients), 1), friction_mu, gpis)

        interp = torch.linspace(0, 1, self.num_samples).cuda().view(1, self.num_samples, 1, 1)
        delta_vector = (self.pregrasp_tip_pose - target_pose).unsqueeze(1).repeat(1, self.num_samples, 1, 1) * interp
        sample_points = target_pose.unsqueeze(1).repeat(1, self.num_samples, 1, 1) + delta_vector
        sample_dist, sample_var = gpis.pred(sample_points.view(-1,4,3))
        sample_prob = (1/torch.sqrt(sample_var))*torch.exp(-sample_dist**2 / (2 * sample_var)) #[num_envs * num_samples, 4, 1]
        sample_prob = sample_prob.view(-1, self.num_samples, 4, 1)
        # Normalize sample probability
        normalization_factor = sample_prob.sum(dim=1) # [num_envs, 4, 1]
        sample_prob = sample_prob / normalization_factor.unsqueeze(1) 
        
        contact_prob = (contact_prob.unsqueeze(2) / normalization_factor).unsqueeze(1) # [num_envs, 1, 4, 1]
        total_prominence = (contact_prob - sample_prob).sum(dim=1).squeeze() # [num_envs, 4]
        total_loss = -total_prominence.sum(dim=1) * 20.0 # NOTE: Variance loss

        self.R, self.t = R, t
        total_loss += (self.pregrasp_weights.unsqueeze(1) * l.view(-1,num_envs)).sum(dim=0)
        self.total_margin = (self.pregrasp_weights.view(-1,1,1) * margin.view(-1, num_envs, 4)).sum(dim=0)
        pre_dist, _ = gpis.pred(self.pregrasp_tip_pose)
        total_loss -= pre_dist.sum(dim=1) * 50.0 # NOTE: Experimental
        # Palm to object distance
        if self.optimize_palm:
            palm_dist,_ = gpis.pred(palm_poses)
            palm_dist = palm_dist
            total_loss += 1/palm_dist # Need to ensure palm is outside the object.
            #print("palm dist:", float(palm_dist.mean()))
        # Should add finger segment penetration loss
        link_pos = self.forward_kinematics(joint_angles, palm_posori, link_names=["link_1.0", "link_2.0", "link_3.0",
                                                                                  "link_5.0", "link_6.0", "link_7.0",
                                                                                  "link_9.0", "link_10.0", "link_11.0"])
        link_dist,_ = gpis.pred(link_pos)
        total_loss += (1/link_dist).sum(dim=1) * 0.01
        
        total_loss += self.compute_collision_loss(joint_angles, palm_posori) #+ torch.abs(hand_dist - 0.05) * 10.0 # NOTE: Experimental
        self.total_loss = total_loss
        loss = total_loss.sum() # TODO: TO BE FINISHED
        loss.backward()
        return loss

    def optimize(self, init_joint_angles, target_pose, compliance, friction_mu, gpis, verbose=True):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        Params:
        joint_angles: [num_envs, num_dofs]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        opt_mask: [num_envs, num_fingers]
        """
        joint_angles = init_joint_angles.clone().requires_grad_(True)
        compliance = compliance.clone().requires_grad_(True)
        params_list = [{"params":joint_angles, "lr":2e-3},
                       {"params":compliance, "lr":0.2}]
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            params_list.append({"params":target_pose, "lr":2e-3})
        
        palm_poses = self.palm_offset[:,:3].clone().requires_grad_(self.optimize_palm)
        palm_oris = self.palm_offset[:,3:].clone().requires_grad_(self.optimize_palm)
        if self.optimize_palm:
            params_list.append({"params":palm_poses, "lr":1e-3})
            params_list.append({"params":palm_oris, "lr":1e-3})

        #lbfgs_params_list = [entry["params"] for entry in params_list]
        #self.optim = torch.optim.LBFGS(lbfgs_params_list, lr=0.1)
        self.optim = torch.optim.AdamW(params_list)

        num_envs = init_joint_angles.shape[0]
        opt_joint_angle = init_joint_angles.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(num_envs).double().cuda()
        opt_margin = torch.zeros(num_envs, 4).double().cuda()
        opt_palm_poses = self.palm_offset.clone()
        opt_R, opt_t = torch.zeros(num_envs, 3, 3).double().cuda(), torch.zeros(num_envs, 3).double().cuda()
        for s in range(self.num_iters):
            #pregrasp_tip_pose, hand_root_pose = self.forward_kinematics(joint_angles, compute_hand_root=True)
            if isinstance(self.optim, torch.optim.LBFGS):
                self.optim.step(partial(self.closure, joint_angles=joint_angles,
                                                      compliance=compliance, 
                                                      target_pose=target_pose, 
                                                      palm_poses=palm_poses, 
                                                      palm_oris=palm_oris, 
                                                      friction_mu=friction_mu, 
                                                      gpis=gpis, num_envs=num_envs))
            else:
                loss = self.closure(joint_angles, compliance, target_pose, palm_poses, palm_oris, friction_mu, gpis, num_envs)
            if verbose:
                print(f"Step {s} Loss:",float(self.total_loss.mean()))
            if torch.isnan(self.total_loss.sum()):
                print("NaN detected:", self.pregrasp_tip_pose, self.total_margin)
            with torch.no_grad():
                update_flag = self.total_loss < opt_value
                if update_flag.sum() and s>20:
                    opt_value[update_flag] = self.total_loss[update_flag].clone()
                    opt_margin[update_flag] = self.total_margin[update_flag].clone()
                    opt_joint_angle[update_flag] = joint_angles[update_flag].clone()
                    opt_target_pose[update_flag] = target_pose[update_flag].clone()
                    opt_compliance[update_flag] = compliance[update_flag].clone()
                    opt_palm_poses[update_flag] = torch.hstack([palm_poses, palm_oris])[update_flag].clone()
                    opt_R[update_flag] = self.R[update_flag].clone()
                    opt_t[update_flag] = self.t[update_flag].clone()
            if not isinstance(self.optim, torch.optim.LBFGS):
                self.optim.step()
            with torch.no_grad(): # apply bounding box constraints
                compliance.clamp_(min=80.0) # prevent negative compliance
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                #palm_poses.clamp_(min=self.palm_offset[:,:3]-0.05, max=self.palm_offset[:,:3]+0.05)
                #palm_oris.clamp_(min=self.palm_offset[:,3:]-1, max=self.palm_offset[:,3:]+1)
                
        print("Margin:",opt_margin)
        return opt_joint_angle, opt_compliance, opt_target_pose, opt_palm_poses, opt_margin, opt_R, opt_t
            

optimizers = {
    "kingpis": KinGPISGraspOptimizer,
    "prob": ProbabilisticGraspOptimizer,
    "wc": WCKinGPISGraspOptimizer,
    "sdf": SDFGraspOptimizer,
    "gpis": GPISGraspOptimizer
}
if __name__ == "__main__":
    import time
    from argparse import ArgumentParser
    import json

    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="prob")
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--use_config", action="store_true", default=False)
    parser.add_argument("--mass", type=float, default=0.5)
    parser.add_argument("--com_x", type=float, default=0.0)
    parser.add_argument("--com_y", type=float, default=0.0)
    parser.add_argument("--com_z", type=float, default=0.0)
    parser.add_argument("--friction", type=float, default=0.5)
    parser.add_argument("--wrist_x", type=float, default=0.0)
    parser.add_argument("--wrist_y", type=float, default=0.0)
    parser.add_argument("--wrist_z", type=float, default=0.0)
    parser.add_argument("--uncertainty", type=float, default=20.0)
    parser.add_argument("--vis_gpis", action="store_true", default=False)
    args = parser.parse_args()

    if args.use_config:
        config = json.load(open(f"assets/{args.exp_name}/config.json"))
        args.wrist_x = config["wrist_x"]
        args.wrist_y = config["wrist_y"]
        args.wrist_z = config["wrist_z"]
        args.mass = config["mass"]
        args.com_x = config["com"][0]
        args.com_y = config["com"][1]
        args.com_z = config["com"][2]
        args.friction = config["friction"]
        args.floor_offset = config["floor_offset"]
        args.wrist_z -= args.floor_offset

    if args.exp_name not in ["realsense", "pvd"]:
        mesh = o3d.io.read_triangle_mesh(f"assets/{args.exp_name}/{args.exp_name}.obj") # TODO: Unify other model
        pcd = mesh.sample_points_poisson_disk(4096)
        center = pcd.get_oriented_bounding_box().get_center()
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
    elif args.exp_name == "realsense":
        pcd = o3d.io.read_point_cloud("../ComplianceDexSensor/obj_cropped.ply")
        #center = pcd.get_oriented_bounding_box().get_center()
        center = pcd.get_axis_aligned_bounding_box().get_center()
        print("AABB:", pcd.get_axis_aligned_bounding_box(), center)
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
    elif args.exp_name == "pvd":
        data = np.load("../ComplianceDexSensor/completed_pcd.npz") # Should be in world frame
        obs_points = data["observed"]
        gen_points = data["completed"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack([obs_points, gen_points]))
        center = pcd.get_oriented_bounding_box().get_center()
        WRIST_OFFSET[:,0] += center[0]
        WRIST_OFFSET[:,1] += center[1]
        WRIST_OFFSET[:,2] += 2 * center[2]
    
    # GPIS formulation
    gpis = GPIS(0.08,1) # Extremely sensitive to kernel length scale
    if args.exp_name in ["realsense", "pvd"]:
        if args.exp_name == "realsense":
            pcd_simple = pcd.farthest_point_down_sample(200)
            points = np.asarray(pcd_simple.points)
            points = torch.tensor(points).cuda().double()
            data_noise = [0.005] * len(points)
        else:
            obs_points = obs_points[:100] # Need subsampling observed points
            gen_points = gen_points[:100]
            points = np.vstack([obs_points, gen_points])
            print("Num points:",len(points)) 
            points = torch.tensor(points).cuda().double()
            data_noise = [0.005] * len(obs_points) + [0.01] * len(gen_points)
        weights = torch.rand(50,len(points)).cuda().double()
        weights = torch.softmax(weights * 20, dim=1)
        internal_points = weights @ points
        bound = 0.1
        externel_points = torch.tensor([[-bound, -bound, -bound], [bound, -bound, -bound], [-bound, bound, -bound],[bound, bound, -bound],
                                    [-bound, -bound, bound], [bound, -bound, bound], [-bound, bound, bound],[bound, bound, bound],
                                    [-bound,0., 0.], [0., -bound, 0.], [bound, 0., 0.], [0., bound, 0],
                                    [0., 0., bound], [0., 0., -bound]]).double().cuda()
        externel_points += torch.from_numpy(center).cuda().double()
        y = torch.vstack([bound * torch.ones_like(externel_points[:,0]).cuda().view(-1,1),
                      torch.zeros_like(points[:,0]).cuda().view(-1,1),
                     -bound * torch.ones_like(internal_points[:,0]).cuda().view(-1,1)])
        gpis.fit(torch.vstack([externel_points, points, internal_points]), y,
                 noise = torch.tensor([0.2] * len(externel_points)+
                                      [0.005] * len(points) +
                                      [0.05] * len(internal_points)).double().cuda())
        if args.vis_gpis:
            test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data([-bound+center[0],-bound+center[1],-bound+center[2]],
                                                                                [bound+center[0],bound+center[1],bound+center[2]],steps=100)
            plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=bound, vmin=-bound)
            plt.show()
            vis_points, vis_normals = gpis.topcd(test_mean, test_normal, [-bound+center[0],-bound+center[1],-bound+center[2]],[bound+center[0],bound+center[1],bound+center[2]],steps=100)
            fitted_pcd = o3d.geometry.PointCloud()
            fitted_pcd.points = o3d.utility.Vector3dVector(vis_points)
            fitted_pcd.normals = o3d.utility.Vector3dVector(vis_normals)
            o3d.visualization.draw_geometries([fitted_pcd, pcd])
            np.savez(f"gpis_states/{args.exp_name}_gpis.npz", mean=test_mean, var=test_var, normal=test_normal, ub=ub, lb=lb)
    else:
        gpis.load_state_data(f"{args.exp_name}_state")
    
    init_tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).double().cuda()
    init_joint_angles = torch.tensor(robot_configs[args.hand]["ref_q"].tolist()).unsqueeze(0).double().cuda()
    # rand_n = torch.rand(4,1)
    # target_pose = rand_n * torch.tensor(FINGERTIP_LB).view(-1,3) + (1 - rand_n) * torch.tensor(FINGERTIP_UB).view(-1,3).double()
    # target_pose = 0.2 * target_pose.unsqueeze(0).cuda()
    # target_pose = torch.from_numpy(center).unsqueeze(0).cuda().double()
    # target_pose = target_pose.repeat(1,4,1)
    if args.mode == "wc":
        compliance = torch.tensor([[200.0,200.0,200.0,400.0]]).cuda()
    else:
        compliance = torch.tensor([[80.0,80.0,80.0,160.0]]).cuda()
    friction_mu = 1
    
    if args.hand == "leap":
        robot_urdf = "pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"
    elif args.hand == "allegro":
        robot_urdf = "pybullet_robot/src/pybullet_robot/robots/allegro_hand/models/allegro_hand_description_left.urdf"
        #robot_urdf = "assets/kuka_allegro/model.urdf"

    WRIST_OFFSET[:,0] += args.wrist_x
    WRIST_OFFSET[:,1] += args.wrist_y
    WRIST_OFFSET[:,2] += args.wrist_z


    if args.mode in ["kingpis", "wc"]:
        grasp_optimizer = optimizers[args.mode](robot_urdf,
                                                ee_link_names=robot_configs[args.hand]["ee_link_name"],
                                                ee_link_offsets=robot_configs[args.hand]["ee_link_offset"].tolist(),
                                                anchor_link_names=robot_configs[args.hand]["collision_links"],
                                                anchor_link_offsets=robot_configs[args.hand]["collision_offsets"].tolist(),
                                                collision_pairs=robot_configs[args.hand]["collision_pairs"],
                                                tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                                                ref_q = robot_configs[args.hand]["ref_q"].tolist(),
                                                optimize_target=True,
                                                optimize_palm=True, # NOTE: Experimental
                                                num_iters=args.num_iters,
                                                palm_offset=WRIST_OFFSET,
                                                uncertainty=args.uncertainty,
                                                # Useless for now
                                                mass=args.mass, com=[args.com_x,args.com_y,args.com_z],
                                                gravity=False)
    elif args.mode == "prob":
        grasp_optimizer = optimizers[args.mode](robot_urdf,
                                                ee_link_names=robot_configs[args.hand]["ee_link_name"],
                                                ee_link_offsets=robot_configs[args.hand]["ee_link_offset"].tolist(),
                                                anchor_link_names=robot_configs[args.hand]["collision_links"],
                                                anchor_link_offsets=robot_configs[args.hand]["collision_offsets"].tolist(),
                                                collision_pairs=robot_configs[args.hand]["collision_pairs"],
                                                tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                                                ref_q = robot_configs[args.hand]["ref_q"].tolist(),
                                                optimize_target=True,
                                                optimize_palm=True, # NOTE: Experimental
                                                num_iters=args.num_iters,
                                                palm_offset=WRIST_OFFSET,
                                                mass=args.mass, com=center[:3],
                                                gravity=False,
                                                uncertainty=args.uncertainty)
    elif args.mode in ["sdf", "gpis"]:
        grasp_optimizer = optimizers[args.mode](tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                                                optimize_target=True, num_iters=args.num_iters,
                                                mass = args.mass, com=[args.com_x,args.com_y,args.com_z],
                                                gravity=False,
                                                uncertainty=args.uncertainty)
    num_guesses = len(WRIST_OFFSET)
    init_joint_angles = init_joint_angles.repeat_interleave(num_guesses,dim=0)
    #target_pose = target_pose.repeat_interleave(num_guesses,dim=0)
    compliance = compliance.repeat_interleave(num_guesses,dim=0)
    debug_tip_pose = grasp_optimizer.forward_kinematics(init_joint_angles, torch.from_numpy(WRIST_OFFSET).cuda())
    target_pose = debug_tip_pose.mean(dim=1, keepdim=True).repeat(1,4,1)
    target_pose = target_pose + (debug_tip_pose - target_pose) * 0.3
    if args.vis_gpis or True:
        for i in range(debug_tip_pose.shape[0]):
            tips, targets = vis_grasp(debug_tip_pose[i], target_pose[i])
            o3d.visualization.draw_geometries([pcd, *tips, *targets])
    if args.mode in ["sdf", "gpis"]:
        opt_tip_pose, opt_compliance, opt_target_pose, success_flag = grasp_optimizer.optimize(init_tip_pose, target_pose, compliance, friction_mu, mesh if args.mode == "sdf" else gpis, verbose=True)
    else:
        
        joint_angles, opt_compliance, opt_target_pose, opt_palm_pose, opt_margin, opt_R, opt_t = grasp_optimizer.optimize(init_joint_angles,target_pose, compliance, friction_mu, gpis, verbose=True)
        opt_tip_pose = grasp_optimizer.forward_kinematics(joint_angles, opt_palm_pose)
    #print("init joint angles:",init_joint_angles)
    # Visualize target and tip pose
    
    pcd.colors = o3d.utility.Vector3dVector(np.array([0.0, 0.0, 1.0] * len(pcd.points)).reshape(-1,3))
    grasp_vis = GraspVisualizer(robot_urdf, pcd)
    for i in range(opt_tip_pose.shape[0]):
        # tips, targets = vis_grasp(opt_tip_pose[i], opt_target_pose[i])
        # o3d.visualization.draw_geometries([pcd, *tips, *targets])
        grasp_vis.visualize_grasp(joint_angles=init_joint_angles[0].detach().cpu().numpy(), 
                                  wrist_pose=opt_palm_pose[i].detach().cpu().numpy(), 
                                  target_pose=opt_target_pose[i].detach().cpu().numpy())
    # After transformation
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    floor = o3d.geometry.TriangleMesh.create_box(width=0.5, height=0.5, depth=0.01).translate([-0.25,-0.25,-0.01])
    idx_list = []
    print(opt_margin)
    for i in range(opt_tip_pose.shape[0]):
        if opt_margin[i].min() > 0.0:
            idx_list.append(i)
        else:
            continue
        tips, targets = vis_grasp(opt_tip_pose[i], opt_target_pose[i])
        o3d.visualization.draw_geometries([pcd, *tips, *targets, coord, floor])
        tip_pose = opt_target_pose[i] + (opt_tip_pose[i] - opt_target_pose[i]) * 0.8
        #check_force_closure(tip_pose, opt_target_pose[i], opt_compliance[i], opt_R[i], opt_t[i], center[:3], args.mass, 10.0 if not args.disable_gravity else 0.0)
        after_tip = (opt_R[i]@opt_tip_pose[i].transpose(0,1)).transpose(0,1)+opt_t[i]
        tips, targets = vis_grasp(after_tip, opt_target_pose[i])
        tf_pcd = copy.deepcopy(pcd)
        tf_pcd.rotate(opt_R[i].cpu().numpy(), center=[0,0,0])
        tf_pcd.translate(opt_t[i].cpu().numpy())
        o3d.visualization.draw_geometries([tf_pcd, *tips, *targets, coord, floor])
    print("Feasible indices:",idx_list)
    if len(idx_list) > 0:
        np.save(f"data/contact_{args.exp_name}.npy", opt_tip_pose.cpu().detach().numpy()[idx_list])
        np.save(f"data/target_{args.exp_name}.npy", opt_target_pose.cpu().detach().numpy()[idx_list])
        np.save(f"data/wrist_{args.exp_name}.npy", opt_palm_pose.cpu().detach().numpy()[idx_list])
        np.save(f"data/compliance_{args.exp_name}.npy", opt_compliance.cpu().detach().numpy()[idx_list])
        np.save(f"data/joint_angle_{args.exp_name}.npy", joint_angles.cpu().detach().numpy()[idx_list])
