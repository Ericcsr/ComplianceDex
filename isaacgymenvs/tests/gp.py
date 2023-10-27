import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, DotProduct, ConstantKernel
x_base = np.linspace(0,1,0).reshape(-1,1)
y_base = np.ones(0,dtype=np.float32).reshape(-1,1)  # Should be dense
noise_scale_base = np.ones(0, dtype=np.float32).reshape(-1,1) * 1.0
x = np.array([0.2,0.3]).reshape(-1,1)
y = np.array([1.0,1.01]).reshape(-1,1)
noise_scale_sample = np.ones(2, dtype=np.float32).reshape(-1,1) * 0.2
alpha = np.vstack([noise_scale_base,noise_scale_sample]).reshape(-1)
x_train = np.vstack([x_base, x])
y_train = np.vstack([y_base, y])

kernel = WhiteKernel(0.2)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=alpha,
                               random_state=0).fit(x_train, y_train)

x_test = np.linspace(0,1).reshape(-1,1)

y_pred, sigma = gpr.predict(x_test, return_std=True)

x_test = x_test.reshape(-1)
y_pred = y_pred.reshape(-1)
sigma = sigma.reshape(-1)
# Initialize plot
f, ax = plt.subplots(1, 1, figsize=(4, 3))

# Get upper and lower confidence bounds
lower, upper = y_pred - sigma, y_pred + sigma
# Plot training data as black stars
ax.plot(x, y, 'k*')
# Plot predictive means as blue line
ax.plot(x_test, y_pred, 'b')
# Shade between the lower and upper confidence bounds
ax.fill_between(x_test, lower, upper, alpha=0.5)
ax.set_ylim([-3, 3])
ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()
