import open3d as o3d
import numpy as np
import torch
from torchsdf import compute_sdf

box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0).translate([-0.5,-0.5,-0.5])
box.scale(0.065, center=[0,0,0])
triangles = np.asarray(box.triangles)
vertices = np.asarray(box.vertices)
faces = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()

points = torch.tensor([[-0.0425,  0.0003,  0.0171],
                        [ 0.0431, -0.0293,  0.0215],
                        [ 0.0963, -0.0950, -0.0334],
                        [ 0.0420,  0.0294,  0.0168]], device='cuda:0') / 0.065

print(compute_sdf(points,faces / 0.065))



