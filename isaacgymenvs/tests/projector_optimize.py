import open3d as o3d
import numpy as np
from torchsdf import compute_sdf
import torch

def optimal_transformation(S1, S2, weight):
    c1 = (weight.diag() @ S1).mean(dim=0)
    c2 = (weight.diag() @ S2).mean(dim=0)
    H = (weight.diag() @ (S1 - c1)).transpose(0,1) @ (weight.diag() @ (S2 - c2))
    U,_,Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.transpose(0,1)
    t = (weight.diag() @ (S2 - (R @ S1.transpose(0,1)).transpose(0,1))).sum(dim=0) / weight.sum()
    return R,t
#@torch.jit.script
def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
    weight = torch.nn.functional.normalize(weight, dim=1, p=1.0)
    weight = weight.unsqueeze(2) # [num_envs, num_points, 1]
    c1 = S1.mean(dim=1).unsqueeze(1) # [num_envs, 3]
    c2 = S2.mean(dim=1).unsqueeze(1)
    H = (weight * (S1 - c1)).transpose(1,2) @ (weight * (S2 - c2))
    U, _, Vh = torch.linalg.svd(H)
    V = Vh.mH
    R = V @ U.transpose(1,2)
    t = (weight * (S2 - (R@S1.transpose(1,2)).transpose(1,2))).sum(dim=1) / weight.sum(dim=1)
    return R, t

# Assume friction is uniform
# Differentiable 
def force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal):
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
    R,t = optimal_transformation_batch(tip_pose, target_pose, compliance) # t: [num_envs, 3]
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose # [num_envs, num_fingers, 3]
    dir_vec = diff_vec / diff_vec.norm(dim=-1).unsqueeze(-1)
    # Rotate local norm to final
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos between
    ang_diff =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    margin = (ang_diff - torch.sqrt(1/(1+torch.tensor(friction_mu)**2))).clamp(min=-0.99)
    # we hope margin to be as large as possible, never below zero
    return torch.log(margin+1).sum(dim=-1)

# sensitive to initial condition 
def project_constraint(tip_pose, target_pose, compliance, opt_mask, friction_mu, object_mesh, num_iters=30, requires_grad=True):
    """
    Params:
    tip_pose: [num_envs, num_fingers, 3]
    target_pose: [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    opt_mask: [num_envs, num_fingers]
    """
    total_opt_mask = opt_mask.sum(dim=-1).bool()
    num_active_env = total_opt_mask.sum()
    tip_pose = tip_pose.clone()
    compliance = compliance.clone()
    triangles = np.asarray(object_mesh.triangles)
    vertices = np.asarray(object_mesh.vertices)
    face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
    var_tip = tip_pose[opt_mask].clone().requires_grad_(requires_grad)
    var_kp = compliance[opt_mask].clone().requires_grad_(requires_grad)
    var_tar = target_pose[opt_mask].unsqueeze(dim=1)
    non_opt_mask = (~opt_mask)
    non_opt_mask[~total_opt_mask] = False
    rest_tip = tip_pose[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1,3)
    rest_kp = compliance[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1)
    rest_tar = tip_pose[non_opt_mask].view(num_active_env,tip_pose.shape[1]-1,3)
    opt_value = torch.inf * torch.ones(num_active_env).cuda()
    opt_tip_pose = tip_pose[opt_mask].clone()
    opt_compliance = compliance[opt_mask].clone()
    optim = torch.optim.RMSprop([var_tip, var_kp],lr=0.001)
    for _ in range(num_iters):
        optim.zero_grad()
        all_tip_pose = torch.cat([rest_tip,var_tip.unsqueeze(dim=1)],dim=1).view(-1,3)
        print(all_tip_pose, var_tip)
        dist,_,current_normal,_ = compute_sdf(all_tip_pose, face_vertices)
        c = - force_eq_reward(torch.cat([rest_tip,var_tip.unsqueeze(dim=1)],dim=1), 
                              torch.cat([rest_tar,var_tar],dim=1), 
                              torch.cat([rest_kp, var_kp.unsqueeze(dim=1)],dim=1), 
                              friction_mu, current_normal.view(num_active_env, tip_pose.shape[1], tip_pose.shape[2]))
        dist = dist.view(num_active_env, tip_pose.shape[1])[:,-1]
        l = c + 100 * dist
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

if __name__ == "__main__":
    import time
    box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).translate([-0.5,-0.5,-0.5])
    box.scale(0.1, center=[0,0,0])
    tip_pose = torch.tensor([[[-0.051, 0.03, -0.04],[0.051,-0.04, 0.03],[0.051,0.0, 0.03],[0.051, 0.04, 0.03]]]*1000).cuda()
    target_pose = torch.tensor([[[-0.03,0., 0.03],[0.03,-0.04, 0.03],[0.03,0., 0.03],[0.03,0.04, 0.03]]]*1000).cuda()
    compliance = torch.tensor([[10.,10.,10.,10.]]* 1000).cuda()
    opt_mask = torch.tensor([[True, False, False, False]]* 1000).cuda()
    opt_mask[1,0] = False
    friction_mu = 0.2
    ts = time.time()
    with torch.no_grad():
        with torch.enable_grad():
            print(project_constraint(tip_pose, target_pose, compliance, opt_mask, friction_mu, box))
    print(time.time() - ts)