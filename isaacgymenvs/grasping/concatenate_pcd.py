import torch
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
from torchsdf import compute_sdf

Rx = Rx = np.array([[1,0,0],[0,0,1],[0,-1,0]])

parser = ArgumentParser()
parser.add_argument("--exp_name", type=str, required=True)
parser.add_argument("--prescale", type=float, default=1.0)
parser.add_argument("--num_samples", type=int, default=32)
args = parser.parse_args()

data = torch.load(f"assets/{args.exp_name}/ours_results.pth", map_location=torch.device("cpu"))['pc'].squeeze().numpy()
delta = np.load(f"assets/{args.exp_name}/delta.npy")
data = data * (1./args.prescale) + delta # By default scale w.r.t origin
camera_pose = Rx @(np.asarray(np.load(f"assets/{args.exp_name}/tf.npz")["t"]) + np.array([0.0, -0.06, -0.15])) + delta
partial_pcd = o3d.io.read_point_cloud(f"assets/{args.exp_name}/{args.exp_name}.ply")
#partial_pcd.paint_uniform_color([1.0, 0.0, 0.0])
completed_points = []

num_points = data.shape[1]

for i in range(data.shape[0]):
    point_samples = np.random.choice(num_points, args.num_samples, replace=False)
    completed_points.append(data[i,point_samples])

completed_points = np.vstack(completed_points)

# remove completed points that occlude observed points
frustum = o3d.geometry.PointCloud()
frustum_points = np.asarray(partial_pcd.points)
frustum.points = o3d.utility.Vector3dVector(frustum_points)
frustum.estimate_normals()
frustum.orient_normals_consistent_tangent_plane(100)
frustum = frustum.farthest_point_down_sample(1024)
radius = 0.005
frustum_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(frustum, o3d.utility.DoubleVector([radius, 2*radius, 4 * radius]))

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(frustum_mesh))
src = np.zeros_like(completed_points)
src[:] = camera_pose
rays = o3d.core.Tensor(np.hstack([src, completed_points - src]), dtype=o3d.core.Dtype.Float32)
ans = scene.cast_rays(rays)
hit_dist = ans['t_hit'].numpy()
mask = hit_dist < 1.0
completed_points = completed_points[mask]

vis_pcd = o3d.geometry.PointCloud()
vis_pcd.points = o3d.utility.Vector3dVector(completed_points)
vis_pcd.paint_uniform_color([0.0, 0.0, 1.0])

observed_points = np.asarray(partial_pcd.points).copy()
np.random.shuffle(observed_points)

np.savez(f"assets/{args.exp_name}/completed_pcd.npz", 
         completed=completed_points, 
         observed = observed_points)
merged_pcd = o3d.geometry.PointCloud()
merged_pcd.points = o3d.utility.Vector3dVector(np.vstack([np.asarray(partial_pcd.points), np.asarray(vis_pcd.points)]))
merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack([np.asarray(partial_pcd.colors), np.asarray(vis_pcd.colors)]))
o3d.visualization.draw_geometries([merged_pcd])
o3d.io.write_point_cloud(f"assets/{args.exp_name}/completed_pcd.ply", merged_pcd)


