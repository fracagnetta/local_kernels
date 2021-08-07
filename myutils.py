import torch

def hypersphere_random_sampler(n_points, input_dim, device):
    x = torch.randn(n_points, input_dim, device=device)
    x /= torch.norm(x, dim=1, keepdim=True)
    return x

def hypercube_random_sampler(n_points, input_dim, device):
    x = torch.rand(n_points, input_dim, device=device)
    return x

def grf_generator(gram, device):
    N = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(len(gram), device=device),  gram)
    y = N.sample()
    return y

def kernel_regression(K_trtr, K_tetr, y_tr, y_te, ridge, device):
    alpha = torch.linalg.inv(K_trtr + ridge * torch.eye(y_tr.size(0), device=device)) @ y_tr
    f = K_tetr @ alpha
    mse = (f - y_te).pow(2).mean()
    return mse
