import open3d as o3d
import numpy as np
from torchsdf import compute_sdf
from gpis import GPIS
import torch

@torch.jit.script
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
    margin = (ang_diff - cos_mu).clamp(min=-0.99)
    print(current_normal)
    # we hope margin to be as large as possible, never below zero
    return torch.log(margin+1).sum(dim=1) + (force * torch.nn.functional.softmin(force)).sum(dim=1)

# sensitive to initial condition 
def project_constraint_sdf(tip_pose, target_pose, compliance, opt_mask, friction_mu, object_mesh, num_iters=100, requires_grad=True):
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
    object_mesh.scale(0.95, center=[0,0,0])
    vertices = np.asarray(object_mesh.vertices)
    face_vertices_deflate = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()
    var_tip = tip_pose[opt_mask].clone().requires_grad_(requires_grad)
    var_kp = compliance[opt_mask].clone().requires_grad_(requires_grad)
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
        
        _,_,current_normal,_ = compute_sdf(all_tip_pose, face_vertices_deflate)
        dist,_,_,_ = compute_sdf(all_tip_pose, face_vertices)
        c = - force_eq_reward(torch.cat([rest_tip,var_tip.unsqueeze(dim=1)],dim=1), 
                              torch.cat([rest_tar,var_tar],dim=1), 
                              torch.cat([rest_kp, var_kp.unsqueeze(dim=1)],dim=1), 
                              friction_mu, current_normal.view(num_active_env, tip_pose.shape[1], tip_pose.shape[2]))
        dist = dist.view(num_active_env, tip_pose.shape[1])[:,-1]
        l = c + 10000 * dist
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

class GraspOptimizer:
    def __init__(self, num_iters, hand_model, object_gpis, num_samples=5, lr=[1e-3, 0.1], isf_barrier=10000, var_cost=10):
        self.num_iters = num_iters
        self.isf_barrier = isf_barrier
        self.var_cost = var_cost
        self.lr = lr
        self.hand_model = hand_model
        self.object_gpis = object_gpis
        self.num_samples = num_samples
        self.samples_p = np.linspace(0,1.0, self.num_samples)

    def check_collision(self,q, hand_base=None):
        """
        check self collision, use default hand base pose if hand_base = None
        :params: q: hand joint angle [num_dofs]
        :params: hand_base: base position respect to the object frame [7]
        :return: collision_flag: whether there are collision
        """
        raise NotImplementedError

    def get_spherical_mapping(self, init_pose, target_pose):
        """
        :params: init_pose: initial tip pose [num_fingers, 3]
        :params: target_pose: target tip pose [num_fingers, 3]
        :return: thetas: rotation along y axis [num_fingers]
        :return: phis: rotation along z axis [num_fingers]
        :return: rs: distance between init and target [num_fingers]
        """
        raise NotImplementedError

    def sample_contacts(self, thetas, phis, rs):
        """
        sample contact configurations between initial and target positions
        :params: thetas: rotation along y axis [num_fingers]
        :params: phis: rotation along z axis [num_fingers]
        :params: rs: distance between init and target [num_fingers]
        :return: contact_configs: [num_samples, num_fingers, 3]
        """
        raise NotImplementedError

    def compute_cost_grad(self, contact_configs, target_pose, compliance, friction_mu):
        """
        compute gradient of all contact configs w.r.t. cost
        :params: contact_configs: [num_samples, num_fingers, 3]
        :params: target_pose: [num_fingers, 3]
        :params: compliance: [num_fingers]
        :friction_mu: float
        :return: contact_configs_grad [num_samples, num_fingers, 3]
        :return: compliance_grad [num_fingers] 
        """
        raise NotImplementedError

    def get_spherical_grad(self, contact_configs_grad, contact_configs, target_pose):
        """
        convert contact_configuration grads to grads in spherical coordiantes,
        weight different gradient component using likelihood value of each samples.
        :params: contact_configs_grad: [num_samples, num_fingers, 3]
        :params: contact_configs: [num_samples, num_fingers, 3]
        :params: target_pose: [num_samples, 3]
        :return: thetas_grad [num_fingers]
        :return: phis_grad [num_fingers]
        """
        raise NotImplementedError

    def get_cartesian_mapping(self, thetas, phis, rs, target_pose, p=[1.0]):
        """
        map spherical coordinate back to cartesian coordinate
        :params: thetas: [num_fingers]
        :params: phis: [num_fingers]
        :params: rs: [num_fingers]
        :params: target_pose: [num_fingers, 3]
        :params: p: list, fractional indexing parameter target_pose + rs * p
        :return: new_init_pose: [num_fingers, 3]
        """
        raise NotImplementedError

    # Should consider kinematic constraints, cannot guarantee a grasp is reachible, but gurantee no kinematic constraint violation
    def optimize_grasp(self, init_pose, target_pose, compliance, friction_mu):
        """
        Optimize grasp with kinematic constraints
        :params: init_pose: [num_fingers, 3]
        :params: target_pose: [num_fingers, 3]
        :params: compliance: [num_fingers]
        :params: friction_mu: float
        :return: new_init_pose: [num_fingers, 3]
        :return: new_target_pose: [num_fingers, 3]
        :return: compliance: [num_fingers]
        """
        compliance = compliance.clone()
        optimizer = torch.optim.RMSprop([{"params":thetas, "lr":self.lr[0]},
                                         {"params":phis, "lr":self.lr[0]},
                                         {"params":compliance, "lr":self.lr[1]}])
        thetas, phis, rs = self.get_spherical_mapping(init_pose, target_pose)
        contact_configs = self.sample_contacts(thetas, phis, rs)
        contact_q = self.solve_ik(contact_configs)
        for i in range(num_iters):
            contact_configs.grad, compliance.grad = self.compute_cost_grad(contact_configs, 
                                                                           target_pose, 
                                                                           compliance, 
                                                                           friction_mu)
            # Check collision and update
            delta_sample_pose = self.lr[0] * contact_configs.grad
            delta_q = self.get_jacobian_pinv(contact_q).T @ delta_sample_pose # need batchify
            collision_flag = self.check_collision(contact_q + delta_q) # [num_samples]
            contact_configs.grad[collision_flag] = 0.0
            thetas.grad, phis.grad = self.get_spherical_grad(contact_configs_grad, target_pose)
            optimizer.step()
        new_init_pose = self.get_cartesian_mapping(thetas, phis, rs)
        return init_pose, target_pose, compliance
            
                

                    

        

            



if __name__ == "__main__":
    import time
    box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).translate([-0.5,-0.5,-0.5])
    box.scale(0.1, center=[0,0,0])
    tip_pose = torch.tensor([[[-0.051, 0., 0.03],[0.03,-0.051, 0.03],[0.03, 0.051, 0.03]]]).cuda()
    target_pose = torch.tensor([[[-0.0,0., 0.03],[0.03,-0.03, 0.03],[0.03,0.03, 0.03]]]).cuda()
    compliance = torch.tensor([[20.,3.,3.]]).cuda()
    opt_mask = torch.tensor([[True, False, False]]).cuda()
    friction_mu = 0.2
    
    # GPIS formulation
    # pcd = box.sample_points_poisson_disk(64)
    # points = torch.from_numpy(np.asarray(pcd.points)).cuda().float()
    # gpis = GPIS()
    # gpis.fit(points, torch.ones_like(points[:,0]).cuda().view(-1,1))
    # ts = time.time()
    # print(project_constraint_gpis(tip_pose, target_pose, compliance, opt_mask, friction_mu, gpis))
    # print(time.time() - ts)

    # SDF formulation
    print(project_constraint_sdf(tip_pose, target_pose, compliance, opt_mask, friction_mu, box))