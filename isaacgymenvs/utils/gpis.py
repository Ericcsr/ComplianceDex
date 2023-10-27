import torch

class GPIS:
    def __init__(self, sigma, bias):
        self.sigma = sigma
        self.bias = bias

    def exponentiated_quadratic(self, xa, xb):
        # L2 distance (Squared Euclidian)
        sq_norm = -0.5 * torch.cdist(xa,xb)**2 / self.sigma**2
        return torch.exp(sq_norm)
    
    def fit(self, X1, y1, noise=0.0):
        self.X1 = X1
        self.y1 = y1 - self.bias
        self.E11 = self.exponentiated_quadratic(X1, X1) + ((noise ** 2) * torch.eye(len(X1)))

    def pred(self, X2, use_grad=False):
        with torch.enable_grad():
            X2 = X2.requires_grad_(use_grad)
            E12 = self.exponentiated_quadratic(self.X1, X2)
            # Solve
            solved = torch.linalg.solve(self.E11, E12).T
            # Compute posterior mean
            mu_2 = solved @ self.y1
            # Compute the posterior covariance
            E22 = self.exponentiated_quadratic(X2, X2)
            E2 = E22 - (solved @ E12)
            if use_grad:
                (mu_2**2).sum().backward()
                normal = X2.grad
                normal = normal / torch.norm(normal, dim=1, keepdim=True)
            else:
                normal = None
            X2.requires_grad_(False)
        return mu_2 + self.bias,  torch.sqrt(torch.diag(E2)), normal