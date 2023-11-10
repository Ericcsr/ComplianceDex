import torch

class GPIS:
    def __init__(self, sigma=0.4, bias=2):
        self.sigma = sigma
        self.bias = bias
        self.fraction = None

    def exponentiated_quadratic(self, xa, xb):
        # L2 distance (Squared Euclidian)
        sq_norm = -0.5 * torch.cdist(xa,xb)**2 / self.sigma**2
        return torch.exp(sq_norm)
    
    def fit(self, X1, y1, noise=0.0):
        self.X1 = X1
        self.y1 = y1 - self.bias
        self.noise = noise
        self.E11 = self.exponentiated_quadratic(X1, X1) + ((self.noise ** 2) * torch.eye(len(X1)).to(X1.device))

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
    
    def pred2(self, X2):
        E12 = self.exponentiated_quadratic(self.X1, X2)
        E11_inv = torch.inverse(self.E11)
        mu_2 = E12.T @ E11_inv @ self.y1
        E22 = self.exponentiated_quadratic(X2, X2)
        E2 = E22 - E12.T @ E11_inv @ E12
        return (mu_2 + self.bias).squeeze(),  torch.sqrt(torch.diag(E2)+1e-6) # prevent nan
    
    # If we only take a subset of X1, we can sample normal from the function
    def compute_normal(self, X2, index=None):
        if index is None:
            idx = torch.arange(len(self.X1)).to(X2.device)
        else:
            idx = torch.tensor(index).to(X2.device)
        with torch.enable_grad():
            X2 = X2.detach().clone()
            X2.requires_grad_(True)
            E12 = self.exponentiated_quadratic(self.X1, X2)
            # Solve
            #solved = torch.linalg.solve(self.E11, E12).T
            E11_inv = torch.inverse(self.E11)
            # Compute posterior mean
            weight = (E11_inv @ self.y1)[idx]
            mu_2 = E12[idx].T @ weight
            mu_2.sum().backward()
            normal = X2.grad
            normal = normal / (torch.norm(normal, dim=1, keepdim=True)+1e-6) # prevent nan when closing to the surface
            if index is None:
                return normal
            else:
                return normal, weight.sum()
        
    def compute_multinormals(self, X2, num_normal_samples):
        """
        :params: num_normal_samples: should be odd int
        :params: X2: [num_test, 3]
        :return: normals: [num_test, num_normal_samples, 3]
        :return: weights: [num_normal_samples]
        """
        if self.fraction is None:
            self.fraction = torch.linspace(0.8,1, num_normal_samples//2)
            self.indices = []
            for i in range(num_normal_samples//2):
                self.indices.append(list(range(int(self.fraction[i] * len(self.X1)))))
            self.indices.append(list(range(len(self.X1))))
            for i in range(num_normal_samples//2):
                self.indices.append(list(range(int(1-self.fraction[-i-1]), len(self.X1))))
        normals = []
        weights = []
        for i in range(num_normal_samples):
            normal, weight = self.compute_normal(X2, self.indices[i])
            normals.append(normal)
            weights.append(weight)
        weights = torch.hstack(weights)
        return torch.stack(normals, dim=1), weights / weights.sum()

        

        
# TODO: Need to visualize GPIS

if __name__ == "__main__":
    import open3d as o3d
    import matplotlib.pyplot as plt
    import numpy as np

    mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate([-0.5,-0.5,-0.5])
    pcd = mesh.sample_points_poisson_disk(64)
    points = torch.from_numpy(np.asarray(pcd.points)).cuda().float()
    gpis = GPIS()
    gpis.fit(points, torch.zeros_like(points[:,0]).cuda().view(-1,1),noise=0.04)
    # test_X = torch.stack(torch.meshgrid(torch.linspace(-1,1,100),torch.linspace(-1,1,100),indexing="xy"),dim=2).cuda()
    # test_z = 0.02 * torch.ones(100,100,1).float().cuda()
    # y_test = torch.zeros(100,100,100).float().cuda()
    # for i in range(100):
    #     #print(test_X.shape, test_z.shape)
    #     x = torch.cat([test_X, i * test_z - 1],dim=2)
    #     print(x[:,:,2])
    #     y_test[i] = gpis.pred(x.view(-1,3))[0].view(100,100)

    # print(y_test.argmin())
    # np.save("gpis.npy", y_test.cpu().numpy())
    # plt.imshow(y_test.cpu().numpy()[50], cmap="seismic", vmax=0.5, vmin=-0.5)
    # plt.show()
    mean1,var1 = gpis.pred(torch.tensor([[-0.6,0, 0],[-0.6,0.0, 0.0]]).cuda())
    mean2,var2 = gpis.pred2(torch.tensor([[-0.6,0, 0],[-0.6,0.0, 0.0]]).cuda())
    auto_normal = gpis.compute_normal(torch.tensor([[-0.6,0, 0],[-0.6,0.0, 0.0]]).cuda())
    #analytical_normal = gpis.analytical_normal(torch.tensor([[-0.6,0, 0],[-0.6,0.0, 0.0]]).cuda())
    normals, weights = gpis.compute_multinormals(torch.tensor([[-0.6,0, 0],[-0.6,0.0, 0.0]]).cuda(), num_normal_samples=5)
    #print(auto_normal, analytical_normal)
    print(mean1, mean2)
    print(var1, var2)
    

