import torch

class GPIS:
    def __init__(self, sigma=0.2, bias=1.0):
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
        E12 = self.exponentiated_quadratic(self.X1, X2)
        # Solve
        solved = torch.linalg.solve(self.E11, E12).T
        # Compute posterior mean
        mu_2 = solved @ self.y1
        # Compute the posterior covariance
        E22 = self.exponentiated_quadratic(X2, X2)
        E2 = E22 - (solved @ E12)
        return mu_2 + self.bias,  torch.sqrt(torch.diag(E2))
    
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