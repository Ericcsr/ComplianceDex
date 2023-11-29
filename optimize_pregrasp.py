import open3d as o3d
import numpy as np
from torchsdf import compute_sdf
from gpis import GPIS
import torch
from differentiable_robot_model.robot_model import DifferentiableRobotModel

EE_OFFSETS = [[0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.04, 0.015],
           [0.0, -0.05, -0.015]]

WRIST_OFFSET = [-0.01, 0.015, 0.12]
REF_Q = [[np.pi/6, -np.pi/9, np.pi/6, np.pi/6,
          np.pi/6, 0.0     , np.pi/6, np.pi/6,
          np.pi/6, np.pi/9 , np.pi/6, np.pi/6,
          np.pi/3, np.pi/4 , np.pi/6, np.pi/6]]

z_margin = 0.02
FINGERTIP_LB = [-0.01, 0.03, -z_margin, -0.01, -0.03, -z_margin, -0.01, -0.08, -z_margin, -0.08, -0.04, -z_margin]
FINGERTIP_UB = [0.08,  0.08,  z_margin - 0.005,  0.08,  0.03,  z_margin - 0.005,  0.08, -0.03,  z_margin- 0.005,  0.01,  0.04,  z_margin- 0.005]

COLLISION_LINKS = ["fingertip","fingertip_2","fingertip_3", "thumb_fingertip",
                   "dip","dip_2","dip_3", "thumb_dip",
                   "pip", "pip_2", "pip_3",
                   "mcp_joint", "mcp_joint_2", "mcp_joint_3"]

COLLISION_OFFSETS = [[0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, -0.05, -0.015],
                     [0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, -0.04, 0.015],[0.0, 0.02, -0.015],
                     [0.01, 0.0, -0.01],[0.01, 0.0, -0.01],[0.01, 0.0, -0.01],
                     [-0.02, 0.04, 0.015],[-0.02, 0.04, 0.015],[-0.02, 0.04, 0.015]]

_COLLISION_PAIRS = [["fingertip", "fingertip_2"], ["fingertip", "fingertip_3"], ["fingertip", "thumb_fingertip"],["fingertip", "dip_2"], ["fingertip", "dip_3"], ["fingertip", "thumb_dip"],
                   ["fingertip_2", "fingertip_3"], ["fingertip_2", "thumb_fingertip"], ["fingertip_2", "dip"], ["fingertip_2", "dip_3"], ["fingertip_2", "thumb_dip"],
                   ["fingertip_3", "thumb_fingertip"], ["fingertip_3", "dip"], ["fingertip_3", "dip_2"], ["fingertip_3", "thumb_dip"],
                   ["thumb_fingertip", "dip"], ["thumb_fingertip", "dip_2"], ["thumb_fingertip", "dip_3"],
                   ["dip", "dip_2"], ["dip", "dip_3"], ["dip", "thumb_dip"], ["dip", "pip_2"], ["dip", "pip_3"], ["dip", "mcp_joint_2"], ["dip", "mcp_joint_3"],
                   ["dip_2", "dip_3"], ["dip_2", "thumb_dip"], ["dip_2", "pip"], ["dip_2", "pip_3"], ["dip_2", "mcp_joint"], ["dip_2", "mcp_joint_3"],
                   ["dip_3", "thumb_dip"], ["dip_3", "pip"], ["dip_3", "pip_2"], ["dip_3", "mcp_joint"], ["dip_3", "mcp_joint_2"],
                   ["pip", "pip_2"], ["pip", "mcp_joint_2"],
                   ["pip_2", "pip_3"], ["pip_2", "mcp_joint"], ["pip_2", "mcp_joint_3"],
                   ["pip_3", "mcp_joint_2"]]
COLLISION_PAIRS = []
for c in _COLLISION_PAIRS:
    COLLISION_PAIRS.append([COLLISION_LINKS.index(c[0]), COLLISION_LINKS.index(c[1])])

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

@torch.jit.script
def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    #weight = torch.nn.functional.normalize(weight, dim=1, p=1.0)
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = S1.mean(dim=1).unsqueeze(1) # [num_envs, 3]
    c2 = S2.mean(dim=1).unsqueeze(1)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H + 1e-6*torch.rand_like(H))
    V = Vh.mH
    R = V @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

# Assume friction is uniform
# Differentiable 
def force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal, mass=0.4, gravity=None, M=1.0, COM=None):
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
    if gravity is not None:
        dummy_tip = torch.zeros(tip_pose.shape[0], 1, 3).cuda()
        dummy_tip[:,0,:] = target_pose.mean(dim=1) if COM is None else COM
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
    margin = (ang_diff - cos_mu).clamp(min=-0.999)
    # if (margin == -0.999).any():
    #     print("Debug:",dir_vec, normal_eq)
    # we hope margin to be as large as possible, never below zero
    force_norm = force.norm(dim=2)
    reward = torch.log(margin+1).sum(dim=1)
    return reward , margin, force_norm

class KinGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names=["fingertip","fingertip_2","fingertip_3", "thumb_fingertip"],
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=[-0.01, 0.015, 0.12],
                 num_iters=1000, optimize_target=False,
                 ref_q=REF_Q):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).cuda()
        self.optimize_target = optimize_target

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names, 
                                                           offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)

    def optimize(self, joint_angles, target_pose, compliance, friction_mu, object_mesh):
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
                                current_normal.view(target_pose.shape))
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
        print(opt_margin, normal)
        return opt_joint_angle, opt_compliance, opt_target_pose

class SDFGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, object_mesh):
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
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]))
            c = -task_reward * 5.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.sqrt(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *(torch.sqrt(tar_dist).view(tip_pose.shape[0], tip_pose.shape[1]) * tar_sign).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost # Encourage target pose to stay inside the object
            l.sum().backward()
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
        print(opt_margin, normal)
        return opt_tip_pose, opt_compliance, opt_target_pose

class GPISGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, optimize_target=False):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, gpis):
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
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]))
            c = -task_reward * 25.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.abs(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *tar_dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            variance_cost = 10.0 * var.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + variance_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()), float(variance_cost.sum()))
            if torch.isnan(l.sum()):
                print("Loss NaN trace:",dist, tip_pose, target_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                compliance.clamp_(min=40.0) # prevent negative compliance
                #target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        print(margin, normal)
        return opt_tip_pose, opt_compliance, opt_target_pose

class KinGPISGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names=["fingertip","fingertip_2","fingertip_3", "thumb_fingertip"],
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, optimize_target=False,
                 ref_q=REF_Q,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB]):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().cuda()
        self.optimize_target = optimize_target
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names,
                                                                offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses.view(-1,3)
    
    def optimize(self, joint_angles, target_pose, compliance, friction_mu, gpis):
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
                                current_normal.view(target_pose.shape))
            c = -task_reward * 5.0
            center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0
            ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
            variance_cost = 20 * torch.log(100 * var).view(tip_pose.shape[0], tip_pose.shape[1])
            dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * tar_dist.view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
            l.sum().backward()
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
        print(opt_margin, normal)
        return opt_joint_angle, opt_compliance, opt_target_pose

class SphericalGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=200, optimize_target=False):
        """
        tip_bounding_box: [lb [num_finger, 3], ub [num_finger, 3]]
        """
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.num_iters = num_iters
        self.optimize_target = optimize_target
        self.device = "cuda:0"

    def get_spherical_mapping(self, init_pose, target_pose, requires_grad=True):
        """
        :params: init_pose: initial tip pose [num_envs, num_fingers, 3]
        :params: target_pose: target tip pose [num_envs, num_fingers, 3]
        :return: thetas: rotation along y axis [num_envs, num_fingers]
        :return: phis: rotation along z axis [num_envs, num_fingers]
        :return: rs: distance between init and target [num_envs, num_fingers]
        """
        delta_pose = init_pose - target_pose
        rs = delta_pose.norm(dim=2)
        thetas = torch.acos(delta_pose[:,:,2] / rs)
        phis = torch.atan2(delta_pose[:,:,1], delta_pose[:,:,0])
        return thetas.requires_grad_(requires_grad), phis.requires_grad_(requires_grad), rs.requires_grad_(requires_grad)
    
    def get_cartesian_mapping(self, thetas, phis, rs, target_pose, p=[1.0]):
        """
        NOTE: Should be fully differentiable
        map spherical coordinate back to cartesian coordinate
        :params: thetas: [num_envs, num_fingers] angle between the vector and z axis
        :params: phis: [num_envs, num_fingers] angle between projection of the vector and x axis
        :params: rs: [num_envs, num_fingers]
        :params: target_pose: [num_envs, num_fingers, 3]
        :params: p: list, fractional indexing parameter target_pose + rs * p
        :return: new_pose: [num_envs, num_samples, num_fingers, 3]
        """
        if p is None:
            num_samples = self.num_samples
            p = self.samples_p.view(1, num_samples, 1)
        else:
            num_samples = len(p)
            p = torch.tensor(p, device=self.device).float().view(1, num_samples, 1)
        rs = rs.unsqueeze(1).repeat(1, num_samples,1)
        thetas = thetas.unsqueeze(1)
        phis = phis.unsqueeze(1)
        new_pose = torch.stack([p * rs * torch.sin(thetas) * torch.cos(phis),
                                p * rs * torch.sin(thetas) * torch.sin(phis),
                                p * rs * torch.cos(thetas)], dim= 3) + target_pose.unsqueeze(1)
        return new_pose.squeeze(1)

    def create_rs_bounding_box(self,thetas, phis, target_pose):
        """
        NOTE: Assume work in torch.no_grad()
        :params: thetas: [num_envs, num_fingers] angle between the vector and z axis
        :params: phis: [num_envs, num_fingers] angle between the vector and x axis
        :params: target_pose: [num_envs, num_fingers, 3]
        :return: rs_ub: [num_envs, num_fingers] bounding boxes
        """
        delta_pose = self.get_cartesian_mapping(thetas, phis, torch.ones_like(thetas).float().cuda(), target_pose)
        dummy_tip_pose = (delta_pose + target_pose).clamp(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        delta_clamped = dummy_tip_pose - target_pose
        return delta_clamped.norm(dim=2) / delta_pose.norm(dim=2)

    def optimize(self, tip_pose, target_pose, compliance, friction_mu, gpis):
        """
        NOTE: scale matters in running optimization, need to normalize the scale
        TODO: Add a penalty term to encourage target pose stay inside the object.
        Params:
        tip_pose: [num_envs, num_fingers, 3]
        target_pose: [num_envs, num_fingers, 3]
        compliance: [num_envs, num_fingers]
        """
        thetas, phis, rs = self.get_spherical_mapping(tip_pose, target_pose)
        compliance = compliance.clone().requires_grad_(True)
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":thetas, "lr":5e-4},
                                         {"params":phis, "lr":5e-4},
                                         {"params":rs, "lr":1e-3},
                                        {"params":target_pose, "lr":1e-3}, 
                                        {"params":compliance, "lr":0.2}])
        else:
            optim = torch.optim.RMSprop([{"params":thetas, "lr":5e-4},
                                         {"params":phis, "lr":5e-4},
                                         {"params":rs, "lr":1e-3},
                                        {"params":compliance, "lr":0.2}])
        opt_thetas = thetas.clone()
        opt_phis = phis.clone()
        opt_rs = rs.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cuda()
        normal = None
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.get_cartesian_mapping(thetas, phis, rs, target_pose).view(-1,3)
            dist, var = gpis.pred(all_tip_pose)
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose)
            task_reward, margin, force_norm = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]))
            c = -task_reward * 5.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=10.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.abs(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 * tar_dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            variance_cost = 10 * var.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            l = c + dist_cost + tar_dist_cost + center_cost + force_cost + variance_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()), float(variance_cost.sum()))
            if torch.isnan(l.sum()):
                print("Loss NaN trace:",dist, tip_pose, target_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_value[update_flag] = l[update_flag]
                    opt_phis[update_flag] = phis[update_flag]
                    opt_thetas[update_flag] = thetas[update_flag]
                    opt_rs[update_flag] = rs[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                compliance.clamp_(min=10.0)
                # Only clip rs
                rs_ub = self.create_rs_bounding_box(thetas, phis, target_pose)
                rs.clamp_(min=torch.zeros_like(rs_ub), max=rs_ub)
        print(margin, normal)
        return self.get_cartesian_mapping(opt_thetas, opt_phis, opt_rs, opt_target_pose), opt_compliance, opt_target_pose

class ProbabilisticGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names=["fingertip","fingertip_2","fingertip_3", "thumb_fingertip"],
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=WRIST_OFFSET,
                 num_iters=1000, optimize_target=False,
                 ref_q=REF_Q,
                 tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB],
                 pregrasp_coefficients = [[0.85,0.85,0.85,0.85],
                                          [0.80,0.80,0.80,0.80],
                                          [0.75,0.75,0.75,0.75]],
                 pregrasp_weights = [0.1,0.8,0.1],
                 anchor_link_names = COLLISION_LINKS,
                 anchor_link_offsets = COLLISION_OFFSETS,
                 collision_pairs = COLLISION_PAIRS,
                 collision_pair_threshold = 0.015):
        self.ref_q = torch.tensor(ref_q).cuda()
        self.robot_model = DifferentiableRobotModel(robot_urdf, device="cuda:0")
        self.num_iters = num_iters
        self.ee_link_names = ee_link_names
        self.ee_link_offsets = ee_link_offsets
        self.palm_offset = torch.tensor(palm_offset).double().cuda()
        self.optimize_target = optimize_target
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.pregrasp_coefficients = torch.tensor(pregrasp_coefficients).cuda()
        self.pregrasp_weights = pregrasp_weights
        self.anchor_link_names = anchor_link_names
        self.anchor_link_offsets = anchor_link_offsets
        collision_pairs = torch.tensor(collision_pairs).long().cuda()
        self.collision_pair_left = collision_pairs[:,0]
        self.collision_pair_right = collision_pairs[:,1]
        self.collision_pair_threshold = collision_pair_threshold

    def forward_kinematics(self, joint_angles):
        """
        :params: joint_angles: [num_envs, num_dofs]
        :return: tip_poses: [num_envs * num_fingers, 3]
        """
        tip_poses = self.robot_model.compute_forward_kinematics(joint_angles, self.ee_link_names,
                                                                offsets=self.ee_link_offsets, recursive=True)[0].view(-1,3) + self.palm_offset
        return tip_poses
    
    def compute_collision_loss(self, joint_angles):
        # Should only work when number of environment is 1
        assert joint_angles.shape[0] == 1
        anchor_pose = self.robot_model.compute_forward_kinematics(joint_angles, 
                                                                  self.anchor_link_names, 
                                                                  offsets=self.anchor_link_offsets, 
                                                                  recursive=True)[0].view(-1,3)
        collision_pair_left = anchor_pose[self.collision_pair_left]
        collision_pair_right = anchor_pose[self.collision_pair_right]
        dist = torch.norm(collision_pair_left - collision_pair_right, dim=1) # [num_collision_pairs]
        mask = dist < self.collision_pair_threshold
        cost = (dist - self.collision_pair_threshold)[mask].sum() * 1000.0
        return cost
    
    def compute_loss(self, all_tip_pose, target_pose, compliance, friction_mu, gpis):
        dist, var = gpis.pred(all_tip_pose)
        tar_dist, _ = gpis.pred(target_pose.view(-1,3))
        current_normal = gpis.compute_normal(all_tip_pose)
        task_reward, margin, force_norm = force_eq_reward(all_tip_pose.view(target_pose.shape), 
                            target_pose, 
                            compliance, 
                            friction_mu, 
                            current_normal.view(target_pose.shape))
        c = -task_reward * 25.0
        center_tip = all_tip_pose.view(target_pose.shape).mean(dim=1)
        center_target = target_pose.mean(dim=1)
        force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=10.0).sum(dim=1) 
        center_cost = (center_tip - center_target).norm(dim=1) * 10.0
        ref_cost = (joint_angles - self.ref_q).norm(dim=1) * 20.0
        variance_cost = 20 * torch.log(100 * var).view(tip_pose.shape[0], tip_pose.shape[1])
        dist_cost = 1000 * torch.abs(dist).view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
        tar_dist_cost = 10 * tar_dist.view(target_pose.shape[0], target_pose.shape[1]).sum(dim=1)
        l = c + dist_cost + tar_dist_cost + center_cost + force_cost + ref_cost + variance_cost.max(dim=1)[0] # Encourage target pose to stay inside the object
        return l, margin

    def optimize(self, joint_angles, target_pose, compliance, friction_mu, gpis):
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
        opt_margin = None
        for s in range(self.num_iters):
            optim.zero_grad()
            pregrasp_tip_pose = self.forward_kinematics(joint_angles).view(target_pose.shape)
            total_loss = 0.0
            total_margin = 0.0
            for i in range(len(self.pregrasp_weights)):
                all_tip_pose = target_pose + self.pregrasp_coefficients[i].view(1,-1, 1) * (pregrasp_tip_pose - target_pose)
                all_tip_pose = all_tip_pose.view(-1,3)
                l, margin = self.compute_loss(all_tip_pose, target_pose, compliance, friction_mu, gpis)
                total_loss += self.pregrasp_weights[i] * l
                total_margin += self.pregrasp_weights[i] * margin
            pre_dist, _ = gpis.pred(pregrasp_tip_pose.view(-1,3))
            total_loss -= pre_dist.view(pregrasp_tip_pose.shape[0], pregrasp_tip_pose.shape[1]).sum(dim=1) * 5.0 # NOTE: Experimental
            total_loss += self.compute_collision_loss(joint_angles) # NOTE: Experimental
            total_loss.sum().backward()
            print(f"Step {s} Loss:",float(total_loss.sum()))
            if torch.isnan(total_loss.sum()):
                print("NaN detected:",pregrasp_tip_pose, margin)
            with torch.no_grad():
                update_flag = total_loss < opt_value
                if update_flag.sum():
                    opt_margin = margin
                    opt_value[update_flag] = total_loss[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                compliance.clamp_(min=40.0) # prevent negative compliance
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        print(opt_margin)
        return opt_joint_angle, opt_compliance, opt_target_pose
            
if __name__ == "__main__":
    import time
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--mode", type=str, default="kingpis")
    parser.add_argument("--wrist_x", type=float, default=0.0)
    parser.add_argument("--wrist_y", type=float, default=0.0)
    parser.add_argument("--wrist_z", type=float, default=0.0)
    args = parser.parse_args()
    # mesh = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1).translate([-0.05,-0.05,-0.05])
    # tip_pose = torch.tensor([[[-0.071, 0., 0.03],[0.03,-0.061, 0.03],[0.03, 0.041, 0.03]]]).cuda()
    # target_pose = torch.tensor([[[-0.0,0., 0.03],[0.03,-0.03, 0.03],[0.03,0.03, 0.03]]]).cuda()
    # compliance = torch.tensor([[20.,3.,3.]]).cuda()

    # mesh = o3d.io.read_triangle_mesh("assets/banana/textured.obj")
    # pcd = o3d.io.read_point_cloud("assets/banana/nontextured.ply")
    # tip_pose = torch.tensor([[[0.00,0.05, 0.01],[0.02,-0.0, -0.01],[0.01,-0.04,0.0],[-0.07,-0.01, 0.01]]]).cuda()
    # target_pose = torch.tensor([[[-0.03, 0.03, 0.0],[-0.03, 0.0, 0.0], [-0.03, -0.03, 0.0],[-0.04, -0.0, 0.0]]]).cuda()
    # compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()

    mesh = o3d.io.read_triangle_mesh(f"assets/{args.exp_name}/{args.exp_name}.obj") # TODO: Unify other model
    pcd = mesh.sample_points_poisson_disk(4096)
    tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).double().cuda()
    joint_angles = torch.tensor([[np.pi/12, -np.pi/9, np.pi/8, np.pi/8,
                np.pi/12, 0.0     , np.pi/8, np.pi/8,
                np.pi/12, np.pi/9 , np.pi/8, np.pi/8,
                np.pi/2.5, np.pi/3 , np.pi/6, np.pi/6]]).double().cuda()
    #target_pose = torch.tensor([[[0.015, 0.04, 0.0],[0.015, -0.0, 0.0], [0.015, -0.03, 0.0],[-0.015, -0.0, 0.0]]]).cuda()
    rand_n = torch.rand(4,1)
    #torch.manual_seed(0)
    target_pose = rand_n * torch.tensor(FINGERTIP_LB).view(-1,3) + (1 - rand_n) * torch.tensor(FINGERTIP_UB).view(-1,3).double()
    target_pose = 0.2 * target_pose.unsqueeze(0).cuda()
    compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()
    # mesh = o3d.io.read_triangle_mesh("assets/hammer/textured.stl")
    # pcd = o3d.io.read_point_cloud("assets/hammer/nontextured.ply")
    # tip_pose = torch.tensor([[[0.04,0.06, 0.02],[0.04,-0.0, -0.01],[0.04,-0.02,0.0],[-0.04,-0.0, 0.01]]]).cuda()
    # target_pose = torch.tensor([[[0.015, 0.04, 0.0],[0.015, -0.0, 0.0], [0.015, -0.04, 0.0],[-0.0, -0.0, 0.0]]]).cuda()
    # compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()
    friction_mu = 0.5
    
    # GPIS formulation
    pcd_gpis = mesh.sample_points_poisson_disk(128)
    points = torch.from_numpy(np.asarray(pcd_gpis.points)).cuda().double()
    #points += torch.randn_like(points) * 0.01
    gpis = GPIS(0.02,0.1) # Extremely sensitive to kernel length scale
    #gpis.fit(points, torch.zeros_like(points[:,0]).cuda().view(-1,1), noise=0.02)
    gpis.load_state_data(f"{args.exp_name}_state")
    # ts = time.time()
    # print(project_constraint_gpis(tip_pose, target_pose, compliance, opt_mask, friction_mu, gpis))
    # print(time.time() - ts)

    # SDF formulation
    #opt_tip_pose, compliance, opt_target_pose = optimize_grasp_sdf(tip_pose, target_pose, compliance, friction_mu, mesh)
    robot_urdf = "pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"
    if args.mode == "kingpis":
        grasp_optimizer = KinGPISGraspOptimizer(robot_urdf, 
                                                tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], 
                                                optimize_target=True, 
                                                num_iters=args.num_iters,
                                                palm_offset=[WRIST_OFFSET[0]+args.wrist_x,WRIST_OFFSET[1]+args.wrist_y,WRIST_OFFSET[2]+args.wrist_z])
        #init_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1,3)
        joint_angles, opt_compliance, opt_target_pose = grasp_optimizer.optimize(joint_angles,target_pose, compliance, friction_mu, gpis)
        opt_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1, 3)
    elif args.mode == "prob":
        grasp_optimizer = ProbabilisticGraspOptimizer(robot_urdf, 
                                                tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], 
                                                optimize_target=True, 
                                                num_iters=args.num_iters,
                                                palm_offset=[WRIST_OFFSET[0]+args.wrist_x,WRIST_OFFSET[1]+args.wrist_y,WRIST_OFFSET[2]+args.wrist_z])
        joint_angles, opt_compliance, opt_target_pose = grasp_optimizer.optimize(joint_angles,target_pose, compliance, friction_mu, gpis)
        opt_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1, 3)
    elif args.mode == "sdf":
        grasp_optimizer = SDFGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True, num_iters=args.num_iters)
    elif args.mode == "gpis":
        grasp_optimizer = GPISGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True, num_iters=args.num_iters)
    elif args.mode == "spherical":
        grasp_optimizer = SphericalGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True, num_iters=args.num_iters)
    
    
    if args.mode == "sdf":
        opt_tip_pose, opt_compliance, opt_target_pose = grasp_optimizer.optimize(tip_pose, target_pose, compliance, friction_mu, mesh)
    elif args.mode not in ["kingpis", "prob"]:
        opt_tip_pose, opt_compliance, opt_target_pose = grasp_optimizer.optimize(tip_pose, target_pose, compliance, friction_mu, gpis)

    # Visualize target and tip pose
    tips, targets = vis_grasp(opt_tip_pose, opt_target_pose)
    o3d.visualization.draw_geometries([pcd, *tips, *targets])
    vis_pcd = o3d.io.read_point_cloud(f"assets/{args.exp_name}/completed_pcd.ply")
    o3d.visualization.draw_geometries([vis_pcd, *tips, *targets])
    print(opt_compliance)
    np.save(f"data/contact_{args.exp_name}.npy", opt_tip_pose.cpu().detach().numpy().squeeze())
    np.save(f"data/target_{args.exp_name}.npy", target_pose.cpu().detach().numpy().squeeze())
    np.save(f"data/compliance_{args.exp_name}.npy", compliance.cpu().detach().numpy().squeeze())
    np.save(f"data/joint_angle_{args.exp_name}.npy", joint_angles.cpu().detach().numpy().squeeze())
    