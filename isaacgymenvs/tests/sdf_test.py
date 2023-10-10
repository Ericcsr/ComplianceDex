import open3d as o3d
import numpy as np
import torch
from torchsdf import compute_sdf

box = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
triangles = np.asarray(box.triangles)
vertices = np.asarray(box.vertices)
faces = torch.from_numpy(vertices[triangles.flatten()].reshape(len(triangles),3,3)).cuda().float()

points = torch.tensor([[-0.5,-0.5,-0.5],
                       [1.5,1.5,1.5]]).cuda().float()

print(compute_sdf(points,faces))



