import torch

class GPIS:
    def __init__(self, sigma=0.4, bias=2):
        self.sigma = sigma
        self.bias = bias

    def exponentiated_quadratic(self, xa, xb):
        # L2 distance (Squared Euclidian)
        sq_norm = -0.5 * torch.cdist(xa,xb)**2 / self.sigma**2
        return torch.exp(sq_norm)
    
    def fit(self, X1, y1, noise=0.0):
        self.X1 = X1
        self.y1 = y1 - self.bias
        self.E11 = self.exponentiated_quadratic(X1, X1) + ((noise ** 2) * torch.eye(len(X1)).to(X1.device))

    def pred(self, X2):
        """
        X2: [num_test, dim]
        """
        E12 = self.exponentiated_quadratic(self.X1, X2)
        # Solve
        solved = torch.linalg.solve(self.E11, E12).T
        # Compute posterior mean
        mu_2 = solved @ self.y1
        # Compute the posterior covariance
        E22 = self.exponentiated_quadratic(X2, X2)
        E2 = E22 - (solved @ E12)
        return (mu_2 + self.bias).squeeze(),  torch.sqrt(torch.diag(E2)+1e-6) # prevent nan
    
    def compute_normal(self, X2):
        with torch.enable_grad():
            X2 = X2.detach().clone()
            X2.requires_grad_(True)
            E12 = self.exponentiated_quadratic(self.X1, X2)
            # Solve
            solved = torch.linalg.solve(self.E11, E12).T
            # Compute posterior mean
            mu_2 = solved @ self.y1
            (mu_2**2).sum().backward()
            normal = X2.grad
            normal = normal / (torch.norm(normal, dim=1, keepdim=True)+1e-6) # prevent nan when closing to the surface
            return -normal
        
# TODO: Need to visualize GPIS

if __name__ == "__main__":
    import open3d as o3d
    import matplotlib.pyplot as plt
    import numpy as np

    mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate([-0.5,-0.5,-0.5])
    pcd = mesh.sample_points_uniformly(64)
    points = torch.from_numpy(np.asarray(pcd.points)).cuda().float()
    gpis = GPIS()
    gpis.fit(points, torch.zeros_like(points[:,0]).cuda().view(-1,1),noise=0.04)
    test_X = torch.stack(torch.meshgrid(torch.linspace(-1,1,100),torch.linspace(-1,1,100),indexing="xy"),dim=2).cuda()
    test_z = 0.02 * torch.ones(100,100,1).float().cuda()
    y_test = torch.zeros(100,100,100).float().cuda()
    for i in range(100):
        #print(test_X.shape, test_z.shape)
        x = torch.cat([test_X, i * test_z - 1],dim=2)
        print(x[:,:,2])
        y_test[i] = gpis.pred(x.view(-1,3))[0].view(100,100)

    print(y_test.argmin())
    np.save("gpis.npy", y_test.cpu().numpy())
    plt.imshow(y_test.cpu().numpy()[50], cmap="seismic", vmax=0.5, vmin=-0.5)
    plt.show()
    

