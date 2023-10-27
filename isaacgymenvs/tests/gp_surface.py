import math
import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
import gpytorch

@torch.jit.script
def linspace(start, stop, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float64, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

# We will use the simplest form of GP model, exact inference
class GPIS(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPIS, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def compute_posterior(self,y_pred, y_stddev):
        return 1/(y_stddev * np.sqrt(2*np.pi)) * torch.exp(-0.5 * (y_pred/y_stddev)**2)
    
    def estimate_normal(self, x):
        """
        Params:
        x: [num_points, 3]
        return:
        normals: always point toward surface [num_points, 3]
        """
        with torch.enable_grad():
            self.train()
            x.requires_grad_(True)
            observed_pred = self.likelihood(self.forward(x))
            (observed_pred.mean**2).sum().backward()
            n = x.grad
            n = n / torch.norm(n, dim=1, keepdim=True)
            x.requires_grad_(False)
            self.eval()
        return n
    
    def pred(self,x):
        return self.likelihood(self.forward(x))

def train(points):
    # Training data is 100 points in [0,1] inclusive regularly spaced
    train_means = points
    # train_stds = 0.2 * torch.ones_like(points, dtype=torch.float64)
    # train_x_distributional = torch.cat([train_means, (train_stds**2).log()],dim=1)
    # True function is sin(2*pi*x) with Gaussian noise
    train_y = torch.zeros(train_means.shape[0], dtype=torch.float64)-1.0

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPIS(train_means, train_y, likelihood)

    training_iter = 350


    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_means)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model, likelihood


# Should similar to predict contact function
def sample_contact(gpis,likelihood,init_pose, target_pose, object_pose=None, steps=50):
    """
        Predict one finger contact for each environment
        Params:
        init_pose: [num_points, 3]
        target_pose: [num_points, 3]
        Return:
        contact_points: [num_points,3]
        contact_normals: [num_points,3]
        """
    #object_pose = object_pose.repeat(1,init_pose.shape[1]).view(-1,7)
    init_pose = init_pose.view(-1,3)
    target_pose = target_pose.view(-1,3)
    #local_init = quat_rotate_inverse(object_pose[:,3:7], init_pose - object_pose[:,0:3])
    #local_tar = quat_rotate_inverse(object_pose[:,3:7], target_pose - object_pose[:,0:3])
    # Create points from local_init to local_tar and query SDF
    #interior_points = linspace(local_init, local_tar, steps).view(-1,3) # [steps, num_envs, 3]
    interior_points = linspace(init_pose, target_pose, steps).view(-1,3) # [steps * num_points, 3]
    #input_points = torch.hstack([interior_points, torch.zeros_like(interior_points,dtype=torch.float64)])
    f_pred = likelihood(gpis(interior_points))
    y_pred = f_pred.mean + 1
    y_stddev = f_pred.stddev
    posterior = gpis.compute_posterior(y_pred, y_stddev).view(steps, init_pose.shape[0]).transpose(0,1)
    posterior = posterior/ posterior.sum(1,keepdim=True)
    # Sample contact points
    interior_points = interior_points.view(steps, init_pose.shape[0], 3).transpose(0,1)
    #sample_idx = torch.multinomial(posterior, 1).squeeze(1)
    sample_idx = torch.argmax(posterior, dim=1)
    print(posterior, y_stddev, y_pred)
    contact_points = interior_points[torch.arange(interior_points.shape[0]),sample_idx]
    # Compute contact normals
    #contact_normals = gpis.estimate_normal(contact_points)
    return contact_points #contact_normals

mesh = o3d.geometry.TriangleMesh.create_box(0.6,0.6,0.6)
mesh.translate([-0.3,-0.3,-0.3])
pcd = mesh.sample_points_poisson_disk(512)
points = torch.from_numpy(np.asarray(pcd.points))

model, likelihood = train(points)

# testing contact sampling:
init_pose = torch.tensor([[-0.4, 0.0, 0.0],[0.1,-0.4,0.0],[0.0, 0.0, 0.4]], dtype=torch.float64)
target_pose = torch.tensor([[-0.2, 0.0, 0.0],[0.1,-0.2,0.0],[0.0, 0.0, 0.2]],dtype=torch.float64)
contact_points = sample_contact(model, likelihood, init_pose, target_pose)
print(contact_points)
#print(contact_normals)







