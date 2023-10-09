import torch

# S1 and S2 are set of paired points
# Find optimal transformation from S1 to S2
# TODO: Accelerate it and write it in batch form
# TODO: Compute constraint violation
# S1, S2, weight should be small, computation is sparse
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
    c1 = (weight * S1).mean(dim=1).unsqueeze(1) # [num_envs, 3]
    c2 = (weight * S2).mean(dim=1).unsqueeze(1)
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

def force_eq_check(tip_pose, target_pose, compliance, friction_mu, current_normal):
    """
    Params:
    tip_pose: world frame [num_envs, num_fingers, 3]
    target_pose: world frame [num_envs, num_fingers, 3]
    compliance: [num_envs, num_fingers]
    friction_mu: scaler
    current_normal: world frame [num_envs, num_fingers, 3]
    
    Returns:
    feasible grasps: [num_envs] bool
    """
    R,t = optimal_transformation_batch(tip_pose, target_pose, compliance) # t: [num_envs, 3]
    tip_pose_eq = (R@tip_pose.transpose(1,2)).transpose(1,2) + t.unsqueeze(1)
    diff_vec = tip_pose_eq - target_pose # [num_envs, num_fingers, 3]
    dir_vec = diff_vec / diff_vec.norm(dim=2).unsqueeze(2)
    # Rotate local norm to final
    normal_eq = (R @ current_normal.transpose(1,2)).transpose(1,2)
    # measure cos between
    ang_sim =  torch.einsum("ijk,ijk->ij",dir_vec, normal_eq)
    feasible_flag = (ang_sim > torch.sqrt(1/(1+torch.tensor(friction_mu)**2))).prod(dim=-1)
    return feasible_flag



def verify(S1, S2, weight, R, t):
    S1 = (R @ S1.transpose(0,1)).transpose(0,1) + t
    F = weight.diag() @ (S2 - S1)
    return F.sum(dim=0)


if __name__ == "__main__":
    dim = 3
    A = torch.rand(1,dim,3,requires_grad=True)
    B = torch.rand(1,dim,3)
    w = torch.rand(1,dim)
    r = force_eq_check(A, B, w, 0.5, A)
    print(r)
