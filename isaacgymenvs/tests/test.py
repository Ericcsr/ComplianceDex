import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.gaussian_process.kernels import Kernel
from sklearn.base import clone

# Custom Kernel
class CustomKernel(Kernel):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    def __call__(self, X, Y=None, eval_gradient=False):
        # Combine RBF and Constant kernels
        return self.k1(X, Y) + self.k2(X, Y)

    def diag(self, X):
        return np.diag(self(X))

    @property
    def is_stationary(self):
        return False  # Modify this as per your requirement

    def clone_with_theta(self, theta):
        cloned = clone(self)
        cloned.k1.theta = theta[:cloned.k1.theta.shape[0]]
        cloned.k2.theta = theta[cloned.k1.theta.shape[0]:]
        return cloned

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, theta):
        self.k1.theta = theta[:self.k1.theta.shape[0]]
        self.k2.theta = theta[self.k1.theta.shape[0]:]

    @property
    def bounds(self):
        return np.vstack((self.k1.bounds, self.k2.bounds))

# Generate synthetic data
np.random.seed(1)
X = np.random.normal(0, 1, 10).reshape(-1, 1)
y = np.sin(X).ravel()

# Define kernels
k1 = 1.0 * RBF(length_scale=1.0)
k2 = C(constant_value=1.0, constant_value_bounds=(1e-3, 1e3))

# Create Gaussian Process Regressor with custom kernel
gp = GaussianProcessRegressor(kernel=CustomKernel(k1, k2))

# Fit the GP model
gp.fit(X, y)

# Predict new points
x_pred = np.linspace(-5, 5, 100).reshape(-1, 1)
y_pred, sigma = gp.predict(x_pred, return_std=True)

# Plot the results
import matplotlib.pyplot as plt
plt.figure()
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x_pred, y_pred, 'k-', label='Predictions')
plt.fill_between(x_pred[:, 0], y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, alpha=0.2)
plt.legend()
plt.show()