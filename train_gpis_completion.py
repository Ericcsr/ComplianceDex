import torch
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
from gpis import GPIS
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--num_obs", type=int, default=128)
parser.add_argument("--num_gen", type=int, default=64)
parser.add_argument("--obs_noise", type=float, default=0.005)
parser.add_argument("--gen_noise", type=float, default=0.02)
parser.add_argument("--num_internals", type=int, default=5)
parser.add_argument("--boundary", type=float, default=0.1)
parser.add_argument("--bias", type=float, default=1.0)

args = parser.parse_args()

data = np.load(f"assets/{args.exp_name}/completed_pcd.npz")
observed = torch.tensor(data["observed"][:args.num_obs]).double().cuda()
completed = torch.tensor(data["completed"][:args.num_gen]).double().cuda()

# Prepare auxiliary data
externel_points = torch.tensor([[-args.boundary, -args.boundary, -args.boundary], 
                                [args.boundary, -args.boundary, -args.boundary], 
                                [-args.boundary, args.boundary, -args.boundary],
                                [args.boundary, args.boundary, -args.boundary],
                                [-args.boundary, -args.boundary, args.boundary], 
                                [args.boundary, -args.boundary, args.boundary], 
                                [-args.boundary, args.boundary, args.boundary],
                                [args.boundary, args.boundary, args.boundary],
                                [-args.boundary,0., 0.], 
                                [0., -args.boundary, 0.], 
                                [args.boundary, 0., 0.], 
                                [0., args.boundary, 0],
                                [0., 0., args.boundary], 
                                [0., 0., -args.boundary]]).double().cuda()

weights = torch.rand(args.num_internals, len(observed) + len(completed)).double().cuda()
weights = torch.softmax(weights * 10, dim=1)
internel_points = weights @ torch.vstack([observed, completed])

gpis = GPIS(0.08, args.bias)

y1 = torch.hstack([args.boundary * torch.ones(len(externel_points)),
                   torch.zeros(len(observed) + len(completed)),
                   -args.boundary * torch.ones(len(internel_points))]).view(-1,1).double().cuda()

gpis.fit(X1 = torch.vstack([externel_points,
                            observed,
                            completed,
                            internel_points]),
        y1 = y1,
        noise=torch.tensor([0.3] * len(externel_points)+
                           [args.obs_noise] * len(observed)+
                           [args.gen_noise] * len(completed)+
                           [0.02] * len(internel_points)).double().cuda())

test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data([-args.boundary,-args.boundary,-args.boundary],[args.boundary,args.boundary,args.boundary],steps=100)
np.savez(f"{args.exp_name}_gpis.npz", mean=test_mean, var=test_var, normal=test_normal, ub=ub, lb=lb)
gpis.save_state_data(f"{args.exp_name}_state")
plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=args.boundary, vmin=-args.boundary)
plt.show()

# Visualize Mesh from GPIS
points, normals = gpis.topcd(test_mean, test_normal, [-args.boundary,-args.boundary,-args.boundary],[args.boundary,args.boundary,args.boundary],steps=100)
fitted_pcd = o3d.geometry.PointCloud()
fitted_pcd.points = o3d.utility.Vector3dVector(points)
fitted_pcd.normals = o3d.utility.Vector3dVector(normals)
rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(fitted_pcd)[0]
rec_mesh.compute_vertex_normals()
rec_mesh.compute_triangle_normals()
rec_mesh.paint_uniform_color([0.7, 0.7, 0.7])
o3d.visualization.draw_geometries([fitted_pcd])
o3d.visualization.draw_geometries([rec_mesh])
