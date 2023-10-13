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

def optimal_transformation_batch(S1, S2, weight):
    """
    S1: [num_envs, num_points, 3]
    S2: [num_envs, num_points, 3]
    weight: [num_envs, num_points]
    """
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

# Noticed that current normal will change, according to SDF
# Only optimizing a particular tip_pose and compliance
# Should have a index term
def optimize_reward(tip_pose, target_pose, compliance, opt_mask, friction_mu, object_mesh, num_iters=20, lr=[0.02,0.02]):
    """
    opt_mask: [num_envs, num_fingers]
    """
    triangles = np.asarray(object_mesh.triangles)
    vertices = np.asarray(object_mesh.vertices)
    face_vertices = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
    tip_pose = tip_pose.clone().requires_grad_(True)
    compliance /= compliance.sum(dim=-1)
    compliance = compliance.clone().requires_grad_(True)
    la = torch.rand(tip_pose.shape[0], device=tip_pose.device).requires_grad_(True) # lambda, slack variable
    opt_la = True
    for i in range(num_iters):
        dist,_,current_normal,_ = compute_sdf(tip_pose.view(-1,3), face_vertices)
        c = - force_eq_reward(tip_pose, target_pose, compliance, friction_mu, current_normal.view(tip_pose.shape))
        l = (c + la * dist.view(tip_pose.shape[0], tip_pose.shape[1])[opt_mask]).sum()
        l.backward()
        if i % 4 == 0:
            opt_la = not opt_la
        with torch.no_grad():
            # if opt_la:
            #     la =la - la.grad * lr[1]
            # else:
            tip_pose[opt_mask] = tip_pose[opt_mask] -tip_pose.grad[opt_mask] * lr[0]
            compliance[opt_mask] = compliance[opt_mask] - compliance.grad[opt_mask] * lr[0]
        tip_pose.requires_grad_(True)
        compliance.requires_grad_(True)
        la.requires_grad_(True)
    return tip_pose, compliance, la

if __name__ == "__main__":
    box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).translate([-0.5,-0.5,-0.5])
    box.scale(0.1, center=[0,0,0])
    tip_pose = torch.tensor([[[-0.051, 0.01, 0.03],[0.051,-0.04, 0.03],[0.051,0.0, 0.03],[0.051, 0.04, 0.03]]]).cuda()
    target_pose = torch.tensor([[[-0.03,0., 0.03],[0.03,-0.04, 0.03],[0.03,0., 0.03],[0.03,0.04, 0.03]]]).cuda()
    compliance = torch.tensor([[10.,10.,10.,10.]]).cuda()
    opt_mask = torch.tensor([[True, False, False, False]]).cuda()
    friction_mu = 1.0
    print(optimize_reward(tip_pose, target_pose, compliance, opt_mask, friction_mu, box))