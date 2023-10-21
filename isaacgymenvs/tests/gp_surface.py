import math
import numpy as np
import open3d as o3d
import torch
import gpytorch


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(gpytorch.priors.NormalPrior(loc=0.0,scale=1.0))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.GaussianSymmetrizedKLKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    def compute_posterior(self,y_pred, y_stddev):
        return 1/(y_stddev * torch.sqrt(2*torch.pi)) * torch.exp()

mesh = o3d.geometry.TriangleMesh.create_box(0.6,0.6,0.6)
mesh.translate([-0.3,-0.3,-0.3])
pcd = mesh.sample_points_poisson_disk(100)
points = torch.from_numpy(np.asarray(pcd.points))



# Training data is 100 points in [0,1] inclusive regularly spaced
train_means = points
train_stds = 0.2 * torch.ones_like(points, dtype=torch.float32)
train_x_distributional = torch.cat([train_means, (train_stds**2).log()],dim=1)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.zeros(train_means.shape[0], dtype=torch.float32)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x_distributional, train_y, likelihood)

training_iter = 50


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
    output = model(train_x_distributional)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

