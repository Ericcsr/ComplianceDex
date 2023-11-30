import torch
import numpy as np

class GPIS:
    def __init__(self, sigma=0.6, bias=2, kernel="tps"):
        self.sigma = sigma
        self.bias = bias
        self.fraction = None
        if kernel == "tps":
            self.kernel_function = self.thin_plate_spline
        elif kernel == "rbf":
            self.kernel_function = self.exponentiated_quadratic
        elif kernel == "joint":
            self.kernel_function = self.joint_kernel

    def exponentiated_quadratic(self, xa, xb):
        # L2 distance (Squared Euclidian)
        sq_norm = -0.5 * torch.cdist(xa,xb)**2 / self.sigma**2
        return torch.exp(sq_norm)
    
    def thin_plate_spline(self, xa, xb, R=None):
        # L2 distance (Squared Euclidian)
        if R is None:
            R = self.R
        sq_norm = torch.cdist(xa,xb)
        return 2 * sq_norm**3 - 3 * R * sq_norm**2 + R**3
    
    def joint_kernel(self, xa, xb):
        return 0.3 * self.exponentiated_quadratic(xa, xb) + 0.7 * self.thin_plate_spline(xa, xb)
        

    # noise can also be a vector
    def fit(self, X1, y1, noise=0.0):
        # Find maximum pair wise distance within training data
        if self.kernel_function == self.thin_plate_spline or self.kernel_function == self.joint_kernel:
            self.R = torch.max(torch.cdist(X1, X1))
        self.X1 = X1
        self.y1 = y1 - self.bias
        self.noise = noise
        self.E11 = self.kernel_function(X1, X1) + ((self.noise ** 2) * torch.eye(len(X1)).to(X1.device))

    def pred(self, X2):
        """
        X2: [num_test, dim]
        """
        E12 = self.kernel_function(self.X1, X2)
        # Solve
        solved = torch.linalg.solve(self.E11, E12).T
        # Compute posterior mean
        mu_2 = solved @ self.y1
        # Compute the posterior covariance
        E22 = self.kernel_function(X2, X2)
        E2 = E22 - (solved @ E12)
        #print(E2.diag())
        return (mu_2 + self.bias).squeeze(),  torch.sqrt(torch.abs(torch.diag(E2))) # prevent nan
    
    def pred2(self, X2):
        E12 = self.kernel_function(self.X1, X2)
        E11_inv = torch.inverse(self.E11)
        mu_2 = E12.T @ E11_inv @ self.y1
        E22 = self.kernel_function(X2, X2)
        E2 = E22 - E12.T @ E11_inv @ E12
        return (mu_2 + self.bias).squeeze(),  torch.sqrt(torch.abs(torch.diag(E2))) # prevent nan
    
    # If we only take a subset of X1, we can sample normal from the function
    def compute_normal(self, X2, index=None):
        if index is None:
            idx = torch.arange(len(self.X1)).to(X2.device)
        else:
            idx = torch.tensor(index).to(X2.device)
        with torch.enable_grad():
            X2 = X2.detach().clone()
            X2.requires_grad_(True)
            E12 = self.kernel_function(self.X1, X2)
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
    
    def get_visualization_data(self, lb, ub, steps=100):
        test_X = torch.stack(torch.meshgrid(torch.linspace(lb[0],ub[0],steps),
                                            torch.linspace(lb[1],ub[1],steps),
                                            torch.linspace(lb[2],ub[2],steps),indexing="xy"),dim=3).double().to(self.X1.device) # [steps, steps, steps, 3]
        test_mean, test_var = torch.zeros(steps,steps,steps), torch.zeros(steps,steps,steps)
        test_normals = torch.zeros(steps,steps,steps,3)
        for i in range(steps):
            mean, var = self.pred2(test_X[i].view(-1,3)) # [steps**3]
            test_normals[i] = self.compute_normal(test_X[i].view(-1,3)).view(steps,steps,3)
            test_mean[i] = mean.view(steps,steps)
            test_var[i] = var.view(steps,steps)

        return test_mean.cpu().numpy(), test_var.cpu().numpy(), test_normals.cpu().numpy(), np.asarray(lb), np.asarray(ub)
    
    def topcd(self,test_mean, test_normal, lb, ub, steps=100):
        lb, ub = np.asarray(lb), np.asarray(ub)
        
        if torch.is_tensor(test_mean):
            test_mean = test_mean.cpu().numpy()[:,:,::-1]
        if torch.is_tensor(test_normal):
            test_normal = test_normal.cpu().numpy()[:,:,::-1]
        internal = test_mean < 0.0
        # find every point in internel that is surrounded by at least one points that is not internal
        mask = np.zeros_like(internal)
        for i in range(1,steps-1):
            for j in range(1,steps-1):
                for k in range(1,steps-1):
                    if internal[i,j,k]:
                        if not internal[i-1,j,k] or not internal[i+1,j,k] or not internal[i,j-1,k] or not internal[i,j+1,k] or not internal[i,j,k-1] or not internal[i,j,k+1]:
                            mask[i,j,k] = 1
        # get three index of each masked point
        all_points = np.stack(np.meshgrid(np.linspace(lb[0],ub[0],steps),
                                      np.linspace(lb[1],ub[1],steps),
                                      np.linspace(lb[2],ub[2],steps),indexing="xy"),axis=3)
        normals = test_normal[mask]
        # convert index to pointcloud
        points = all_points[mask]
        return points, normals

        

    
    def save_state_data(self, name="gpis_state"):
        R = self.R.cpu().numpy()
        X1 = self.X1.cpu().numpy()
        y1 = self.y1.cpu().numpy()
        E11 = self.E11.cpu().numpy()
        np.savez(f"gpis_states/{name}.npz", R=R, X1=X1, y1=y1, E11=E11, bias=self.bias)

    def load_state_data(self, name="gpis_state"):
        data = np.load(f"gpis_states/{name}.npz")
        self.R = torch.from_numpy(data["R"]).cuda()
        self.X1 = torch.from_numpy(data["X1"]).cuda()
        self.y1 = torch.from_numpy(data["y1"]).cuda()
        self.E11 = torch.from_numpy(data["E11"]).cuda()
        self.bias = torch.from_numpy(data["bias"]).double().cuda()

        
# TODO: Need to visualize GPIS

if __name__ == "__main__":
    import open3d as o3d
    import matplotlib.pyplot as plt
    import numpy as np

    # mesh = o3d.geometry.TriangleMesh.create_box(1, 1, 1).translate([-0.5,-0.5,-0.5])
    # pcd = mesh.sample_points_poisson_disk(64)
    # points = torch.from_numpy(np.asarray(pcd.points)).cuda().float()
    mesh = o3d.io.read_triangle_mesh("assets/mug/mug.stl")
    mesh2 = o3d.io.read_triangle_mesh("assets/mug2/mug2.stl")
    pcd = mesh.sample_points_poisson_disk(128)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0,1,0]), (len(pcd.points),1)))
    pcd2 = mesh2.sample_points_poisson_disk(512)
    o3d.visualization.draw_geometries([pcd])
    points = np.asarray(pcd.points)
    np.random.shuffle(points)
    points = torch.from_numpy(points).cuda().double()
    points2 = torch.from_numpy(np.asarray(pcd2.points)).cuda().double()
    weights = torch.rand(5,128).cuda().double()
    # normalize dimension 1
    weights = torch.softmax(weights * 10, dim=1)
    print(weights.sum(dim=1))
    
    mask = points[:,0] > 0.0
    mask2 = points2[:,0] > 0.0

    internal_points = weights @ points
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(internal_points.cpu().numpy())
    new_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([1,0,0]), (len(internal_points),1)))

    second_pcd = o3d.geometry.PointCloud()
    second_pcd.points = o3d.utility.Vector3dVector(points[~mask].cpu().numpy() * np.array([0.7,1,1]))
    second_pcd.colors = o3d.utility.Vector3dVector(np.tile(np.asarray([0,0,1]), (len(points[~mask]),1)))

    o3d.visualization.draw_geometries([pcd, new_pcd, second_pcd])
    gpis = GPIS(0.08, 1)
    externel_points = torch.tensor([[-0.1, -0.1, -0.1], [0.1, -0.1, -0.1], [-0.1, 0.1, -0.1],[0.1, 0.1, -0.1],
                                    [-0.1, -0.1, 0.1], [0.1, -0.1, 0.1], [-0.1, 0.1, 0.1],[0.1, 0.1, 0.1],
                                    [-0.1,0., 0.], [0., -0.1, 0.], [0.1, 0., 0.], [0., 0.1, 0],
                                    [0., 0., 0.1], [0., 0., -0.1]]).double().cuda()
    
    # Create mask for partial observed pointcloud
    

    y = torch.vstack([0.1 * torch.ones_like(externel_points[:,0]).cuda().view(-1,1),
                      torch.zeros_like(points[mask][:,0]).cuda().view(-1,1),
                      torch.zeros_like(points[~mask][:,0]).cuda().view(-1,1),
                      torch.zeros_like(points2[~mask2][:,0]).cuda().view(-1,1),
                     -0.1 * torch.ones_like(internal_points[:,0]).cuda().view(-1,1)])
    print(y.shape)
    gpis.fit(torch.vstack([externel_points, 
                           points[mask], 
                           points[~mask],
                           points2[~mask2],
                           internal_points]), 
                           y,noise=torch.tensor([0.3] * len(externel_points)+
                                                [0.005] * len(points[mask]) +  # Observed
                                                [0.02] * len(points[~mask]) +  # Completion 1
                                                [0.02] * len(points2[~mask2]) +  # Completion 2
                                                [0.02] * len(internal_points)).double().cuda()) # Internal points
    test_mean, test_var, test_normal, lb, ub = gpis.get_visualization_data([-0.1,-0.1,-0.1],[0.1,0.1,0.1],steps=100)
    np.savez("gpis_states/dummy_gpis.npz", mean=test_mean, var=test_var, normal=test_normal, ub=ub, lb=lb)
    gpis.save_state_data("dummy_state")
    plt.imshow(test_mean[:,:,50], cmap="seismic", vmax=0.1, vmin=-0.1)
    plt.show()

    points, normals = gpis.topcd(test_mean, test_normal, [-0.1,-0.1,-0.1],[0.1,0.1,0.1],steps=100)
    fitted_pcd = o3d.geometry.PointCloud()
    fitted_pcd.points = o3d.utility.Vector3dVector(points)
    fitted_pcd.normals = o3d.utility.Vector3dVector(normals)
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(fitted_pcd)[0]
    rec_mesh.compute_vertex_normals()
    rec_mesh.compute_triangle_normals()
    rec_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    o3d.visualization.draw_geometries([fitted_pcd])
    o3d.visualization.draw_geometries([rec_mesh])
    
