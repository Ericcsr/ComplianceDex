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

z_margin = 0.02
FINGERTIP_LB = [-0.01, 0.03, -z_margin, -0.01, -0.03, -z_margin, -0.01, -0.08, -z_margin, -0.08, -0.06, -z_margin]
FINGERTIP_UB = [0.08,  0.08,  z_margin,  0.08,  0.03,  z_margin,  0.08, -0.03,  z_margin,  0.01,  0.06,  z_margin]

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
    ang_diff =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    cos_mu = torch.sqrt(1/(1+torch.tensor(friction_mu)**2))
    margin = (ang_diff - cos_mu).clamp(min=-0.999)
    # if (margin == -0.999).any():
    #     print("Debug:",dir_vec, normal_eq)
    # we hope margin to be as large as possible, never below zero
    force_norm = force.norm(dim=2)
    reward = torch.log(margin+1).sum(dim=1)
    return reward , margin, force_norm

# sensitive to initial condition 
def optimize_grasp_sdf(tip_pose, target_pose, compliance, friction_mu, object_mesh, num_iters=200, optimize_target=True):
    """
    NOTE: scale matters in running optimization, need to normalize the scale
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
    if optimize_target:
        target_pose = target_pose.clone().requires_grad_(True)
        optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                     {"params":target_pose, "lr":1e-5}, 
                                     {"params":compliance, "lr":0.2}])
    else:
        optim = torch.optim.RMSprop([{"params":tip_pose, "lr":1e-3},
                                     {"params":compliance, "lr":0.2}])
    opt_tip_pose = tip_pose.clone()
    opt_compliance = compliance.clone()
    opt_target_pose = target_pose.clone()
    opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cuda()
    normal = None
    for _ in range(num_iters):
        optim.zero_grad()
        all_tip_pose = tip_pose.view(-1,3)
        _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
        dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
        # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
        current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
        current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
        r, margin = force_eq_reward(tip_pose, 
                              target_pose, 
                              compliance, 
                              friction_mu, 
                              current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]))
        c = -r
        dist = dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
        l = c + 1000 * torch.sqrt(dist)
        l.sum().backward()
        print("Loss:",float(l.sum()))
        if torch.isnan(l.sum()):
            print(dist, tip_pose, margin)
        with torch.no_grad():
            update_flag = l < opt_value
            if update_flag.sum():
                normal = current_normal
                opt_value[update_flag] = l[update_flag]
                opt_tip_pose[update_flag] = tip_pose[update_flag]
                opt_target_pose[update_flag] = target_pose[update_flag]
                opt_compliance[update_flag] = compliance[update_flag]
        optim.step()
    print(margin, normal)
    return opt_tip_pose, opt_compliance, opt_target_pose

def optimize_grasp_gpis(tip_pose, target_pose, compliance, opt_mask, friction_mu, gpis, num_iters=10):
    total_opt_mask = opt_mask.sum(dim=-1).bool()
    num_active_env = total_opt_mask.sum()
    tip_pose = tip_pose.clone()
    compliance = compliance.clone()
    var_tip = tip_pose[opt_mask].clone().requires_grad_(True)
    var_kp = compliance[opt_mask].clone().requires_grad_(True)
    var_tar = target_pose[opt_mask].unsqueeze(dim=1)
    non_opt_mask = (~opt_mask)
    non_opt_mask[~total_opt_mask] = False
    rest_tip = tip_pose[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1,3)
    rest_kp = compliance[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1)
    rest_tar = target_pose[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1,3)
    opt_value = torch.inf * torch.ones(num_active_env).cuda()
    opt_tip_pose = tip_pose[opt_mask].clone()
    opt_compliance = compliance[opt_mask].clone()
    optim = torch.optim.RMSprop([{"params":var_tip, "lr":1e-3}, 
                                 {"params":var_kp, "lr":0.1}])
    for _ in range(num_iters):
        optim.zero_grad()
        all_tip_pose = torch.cat([rest_tip,var_tip.unsqueeze(dim=1)],dim=1).view(-1,3)
        dist,var = gpis.pred(all_tip_pose)
        current_normal = gpis.compute_normal(all_tip_pose)
        c = - force_eq_reward(torch.cat([rest_tip,var_tip.unsqueeze(dim=1)],dim=1), 
                              torch.cat([rest_tar,var_tar],dim=1), 
                              torch.cat([rest_kp, var_kp.unsqueeze(dim=1)],dim=1), 
                              friction_mu, current_normal.view(num_active_env, tip_pose.shape[1], tip_pose.shape[2]))
        dist = dist.view(num_active_env, tip_pose.shape[1])[:,-1]
        # Should add cost term for variance
        l = c + 10000 * dist**2 + var * 10
        l.sum().backward()
        with torch.no_grad():
            update_flag = l < opt_value
            if update_flag.sum():
                opt_value[update_flag] = l[update_flag]
                opt_tip_pose[update_flag] = var_tip[update_flag]
                opt_compliance[update_flag] = var_kp[update_flag]
        optim.step()
    tip_pose[opt_mask] = var_tip.detach()
    compliance[opt_mask] = var_kp.detach()
    return tip_pose, compliance

class KinGraspOptimizer:
    def __init__(self, 
                 robot_urdf, 
                 ee_link_names=["fingertip","fingertip_2","fingertip_3", "thumb_fingertip"],
                 ee_link_offsets = EE_OFFSETS,
                 palm_offset=[-0.03, 0.015, 0.11],
                 num_iters=1000, optimize_target=False):
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
        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.forward_kinematics(joint_angles)
            _,sign1,current_normal1,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
            dist,sign2,current_normal2,_ = compute_sdf(all_tip_pose, face_vertices)
            # Note: normal direction will flip when tip is inside the object, normal vector at surface is not defined.
            current_normal = 0.5 * sign1.unsqueeze(1) * current_normal1 + 0.5 * sign2.unsqueeze(1) * current_normal2
            current_normal = current_normal / current_normal.norm(dim=1).unsqueeze(1)
            r, margin = force_eq_reward(tip_pose, 
                                target_pose, 
                                compliance, 
                                friction_mu, 
                                current_normal.view(tip_pose.shape[0], tip_pose.shape[1], tip_pose.shape[2]))
            c = -r
            dist = dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            l = c + 1000 * torch.sqrt(dist)
            l.sum().backward()
            print("Loss:",float(l.sum()))
            if torch.isnan(l.sum()):
                print(dist, tip_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value
                if update_flag.sum():
                    normal = current_normal
                    opt_value[update_flag] = l[update_flag]
                    opt_joint_angle[update_flag] = joint_angles[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
        print(margin, normal)
        return opt_joint_angle, opt_compliance, opt_target_pose

class LooseKinGraspOptimizer:
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
                    opt_value[update_flag] = l[update_flag]
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        print(margin, normal)
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
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cuda()
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
            c = -task_reward * 5.0
            center_tip = tip_pose.mean(dim=1)
            center_target = target_pose.mean(dim=1)
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).sum(dim=1)
            center_cost = (center_tip - center_target).norm(dim=1) * 10.0

            dist_cost = 1000 * torch.abs(dist).view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
            tar_dist_cost = 10 *tar_dist.view(tip_pose.shape[0], tip_pose.shape[1]).sum(dim=1)
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
                    opt_tip_pose[update_flag] = tip_pose[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                #target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        print(margin, normal)
        return opt_tip_pose, opt_compliance, opt_target_pose

# Should revise this algorithm with LooseKinGraspOptimizer
# need to systematically setting magical weights
class ProbabilisticGraspOptimizer:
    def __init__(self, tip_bounding_box, num_iters=2000, num_samples=5, lr=[1e-3, 0.2, 1e-3], isf_barrier=1000, var_cost=10, optimize_target=False, device="cuda:0"):
        self.num_iters = num_iters
        self.tip_bounding_box = [torch.tensor(tip_bounding_box[0]).cuda().view(-1,3), torch.tensor(tip_bounding_box[1]).cuda().view(-1,3)]
        self.isf_barrier = isf_barrier
        self.var_cost = var_cost
        self.lr = lr
        self.num_samples = num_samples
        self.optimize_target = optimize_target
        self.device = device
        self.samples_p = torch.linspace(0, 1.0, self.num_samples).to(device)

    def get_spherical_mapping(self, init_pose, target_pose):
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
        return thetas, phis, rs
        
    def gaussian_likelihood(self, mean, std):
        return (1/std) * torch.exp(-0.5 * mean / std**2)

    def get_cartesian_mapping(self, thetas, phis, rs, target_pose, p=None):
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

    # Should consider kinematic constraints, cannot guarantee a grasp is reachible, but gurantee no kinematic constraint violation
    def optimize(self, init_pose, target_pose, compliance, friction_mu, gpis):
        """
        Optimize grasp with kinematic constraints
        :params: init_pose: [num_envs, num_fingers, 3]
        :params: target_pose: [num_envs, num_fingers, 3]
        :params: compliance: [num_envs, num_fingers]
        :params: friction_mu: float
        :return: new_init_pose: [num_envs, num_fingers, 3]
        :return: new_target_pose: [num_envs, num_fingers, 3]
        :return: compliance: [num_envs, num_fingers]
        """
        init_pose = init_pose.clone()
        compliance = compliance.clone()
        thetas, phis, rs = self.get_spherical_mapping(init_pose, target_pose)
        rs = torch.ones_like(thetas).float().to(thetas.device)
        if self.optimize_target:
            target_pose = target_pose.clone().requires_grad_(True)
            optim = torch.optim.RMSprop([{"params":thetas, "lr":self.lr[0]},
                                         {"params":phis, "lr":self.lr[0]},
                                         {"params":target_pose, "lr":self.lr[2]}, 
                                         {"params":compliance, "lr":self.lr[1]}])
        else:
            optim = torch.optim.RMSprop([{"params":thetas, "lr":self.lr[0]},
                                         {"params":phis, "lr":self.lr[0]},
                                         {"params":compliance, "lr":0.2}])
        opt_theta = thetas.clone()
        opt_phi = phis.clone()
        opt_compliance = compliance.clone()
        opt_target_pose = target_pose.clone()
        opt_value = torch.inf * torch.ones(tip_pose.shape[0]).cuda()
        normal = None
        # prepare some dimensional data
        num_envs, num_fingers = init_pose.shape[0], init_pose.shape[1]

        for _ in range(self.num_iters):
            optim.zero_grad()
            all_tip_pose = self.get_cartesian_mapping(thetas, phis, rs, target_pose).view(-1,3)
            dist, var = gpis.pred(all_tip_pose) 
            tar_dist, _ = gpis.pred(target_pose.view(-1,3))
            current_normal = gpis.compute_normal(all_tip_pose) # [num_envs * num_samples * num_fingers, 3]
            # posterior weight, should normalize likelihood value over different samples.
            likelihood = self.gaussian_likelihood(dist, var).view(num_envs, self.num_samples, num_fingers) # [num_envs * num_sample * num_fingers]
            posterior_weight = likelihood / likelihood.sum(dim=1).unsqueeze(1) # [num_envs, num_samples, num_fingers]
            
            # TODO: Need to check tensor layout
            extended_target_pose = target_pose.unsqueeze(1).repeat(1, self.num_samples, 1, 1).view(num_envs * self.num_samples, num_fingers, 3)
            _, margin, force_norm = force_eq_reward(
                                all_tip_pose.view(num_envs * self.num_samples,
                                                  num_fingers, 3),
                                extended_target_pose, 
                                compliance.repeat(self.num_samples,1), 
                                friction_mu, 
                                current_normal.view(num_envs * self.num_samples, 
                                                    num_fingers, 3))
            
            center_tip = all_tip_pose.view(num_envs, self.num_samples, num_fingers, 3).mean(dim=2)
            center_target = extended_target_pose.mean(dim=2)
            center_cost = (center_tip - center_target).norm(dim=2) * 10.0 # [num_envs, num_samples]

            task_cost = -torch.log(margin+1).view(num_envs, self.num_samples, num_fingers) * 5.0
            force_cost = -(force_norm * torch.nn.functional.softmin(force_norm,dim=1)).clamp(max=1.0).view(num_envs, self.num_samples, num_fingers)
            dist_cost = self.isf_barrier * torch.abs(dist).view(num_envs , self.num_samples, num_fingers)
            variance_cost = self.var_cost * var.view(num_envs , self.num_samples, num_fingers)
            print(posterior_weight.shape, task_cost.shape, dist_cost.shape, force_cost.shape, variance_cost.shape)
            fingerwise_cost = posterior_weight * (task_cost + dist_cost + force_cost + variance_cost)
            tar_dist_cost = 10 * tar_dist.view(num_envs , num_fingers).sum(dim=1)
            # need to weight loss from each samples with their total likelihood
            l = fingerwise_cost.view(-1,num_fingers).sum(dim=1) + center_cost.sum(dim=1) + tar_dist_cost # Encourage target pose to stay inside the object
            l.sum().backward()
            print("Loss:",float(l.sum()), float(force_cost.sum()), float(center_cost.sum()), float(dist_cost.sum()), float(tar_dist_cost.sum()), float(variance_cost.sum()))
            if torch.isnan(l.sum()):
                print("Loss NaN trace:",dist, tip_pose, target_pose, margin)
            with torch.no_grad():
                update_flag = l < opt_value # mask on env level
                if update_flag.sum():
                    normal = current_normal
                    opt_value[update_flag] = l[update_flag]
                    opt_theta[update_flag] = thetas[update_flag]
                    opt_phi[update_flag] = phis[update_flag]
                    opt_target_pose[update_flag] = target_pose[update_flag]
                    opt_compliance[update_flag] = compliance[update_flag]
            optim.step()
            with torch.no_grad(): # apply bounding box constraints
                #tip_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
                compliance.clamp_(min=0.001) # prevent negative compliance
                #target_pose.clamp_(min=self.tip_bounding_box[0], max=self.tip_bounding_box[1])
        print(margin, normal)
        return self.get_cartesian_mapping(opt_theta, opt_phi, rs, opt_target_pose, [1.0]), opt_compliance, opt_target_pose
            
if __name__ == "__main__":
    import time

    # mesh = o3d.geometry.TriangleMesh.create_box(0.1, 0.1, 0.1).translate([-0.05,-0.05,-0.05])
    # tip_pose = torch.tensor([[[-0.071, 0., 0.03],[0.03,-0.061, 0.03],[0.03, 0.041, 0.03]]]).cuda()
    # target_pose = torch.tensor([[[-0.0,0., 0.03],[0.03,-0.03, 0.03],[0.03,0.03, 0.03]]]).cuda()
    # compliance = torch.tensor([[20.,3.,3.]]).cuda()

    mesh = o3d.io.read_triangle_mesh("assets/banana/textured.obj")
    pcd = o3d.io.read_point_cloud("assets/banana/nontextured.ply")
    tip_pose = torch.tensor([[[0.00,0.05, 0.01],[0.02,-0.0, -0.01],[0.01,-0.04,0.0],[-0.07,-0.01, 0.01]]]).cuda()
    # target_pose = torch.tensor([[[-0.03, 0.03, 0.0],[-0.03, 0.0, 0.0], [-0.03, -0.03, 0.0],[-0.04, -0.0, 0.0]]]).cuda()
    # compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()

    # mesh = o3d.io.read_triangle_mesh("assets/lego/textured_cvx.stl")
    # pcd = o3d.io.read_point_cloud("assets/lego/nontextured.ply")
    # tip_pose = torch.tensor([[[0.05,0.05, 0.02],[0.06,-0.0, -0.01],[0.03,-0.04,0.0],[-0.07,-0.01, 0.02]]]).cuda()
    # target_pose = torch.tensor([[[0.015, 0.04, 0.0],[0.015, -0.0, 0.0], [0.015, -0.03, 0.0],[-0.015, -0.0, 0.0]]]).cuda()
    rand_n = torch.rand(4,1)
    target_pose = rand_n * torch.tensor(FINGERTIP_LB).view(-1,3) + (1 - rand_n) * torch.tensor(FINGERTIP_UB).view(-1,3)
    target_pose = 0.5 * target_pose.unsqueeze(0).cuda()
    compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()
    # mesh = o3d.io.read_triangle_mesh("assets/hammer/textured.stl")
    # pcd = o3d.io.read_point_cloud("assets/hammer/nontextured.ply")
    # tip_pose = torch.tensor([[[0.04,0.06, 0.02],[0.04,-0.0, -0.01],[0.04,-0.02,0.0],[-0.04,-0.0, 0.01]]]).cuda()
    # target_pose = torch.tensor([[[0.015, 0.04, 0.0],[0.015, -0.0, 0.0], [0.015, -0.04, 0.0],[-0.0, -0.0, 0.0]]]).cuda()
    # compliance = torch.tensor([[10.0,10.0,10.0,20.0]]).cuda()
    friction_mu = 0.5
    
    # GPIS formulation
    pcd_gpis = mesh.sample_points_poisson_disk(64)
    points = torch.from_numpy(np.asarray(pcd_gpis.points)).cuda().float()
    gpis = GPIS(0.02,0.1)
    gpis.fit(points, torch.zeros_like(points[:,0]).cuda().view(-1,1))
    # ts = time.time()
    # print(project_constraint_gpis(tip_pose, target_pose, compliance, opt_mask, friction_mu, gpis))
    # print(time.time() - ts)

    # SDF formulation
    #opt_tip_pose, compliance, opt_target_pose = optimize_grasp_sdf(tip_pose, target_pose, compliance, friction_mu, mesh)
    #robot_urdf = "pybullet_robot/src/pybullet_robot/robots/leap_hand/assets/leap_hand/robot.urdf"
    # grasp_optimizer = KinGraspOptimizer(robot_urdf)
    # init_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1,3)
    # joint_angles, opt_compliance, opt_target_pose = grasp_optimizer.optimize(joint_angles,target_pose, compliance, friction_mu, mesh)
    # opt_tip_pose = grasp_optimizer.forward_kinematics(joint_angles).view(1,-1,3)
    
    #grasp_optimizer = LooseKinGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True)
    #grasp_optimizer = GPISGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True)
    grasp_optimizer = ProbabilisticGraspOptimizer(tip_bounding_box=[FINGERTIP_LB, FINGERTIP_UB], optimize_target=True)
    opt_tip_pose, compliance, opt_target_pose = grasp_optimizer.optimize(tip_pose, target_pose, compliance, friction_mu, gpis)

    # Visualize target and tip pose
    tips, targets = vis_grasp(opt_tip_pose, opt_target_pose)
    o3d.visualization.draw_geometries([pcd, *tips, *targets])
    print(compliance)
    np.save("data/contact_banana_.npy", opt_tip_pose.cpu().detach().numpy().squeeze())
    np.save("data/target_banana_.npy", target_pose.cpu().detach().numpy().squeeze())
    np.save("data/compliance_banana_.npy", compliance.cpu().detach().numpy().squeeze())
    