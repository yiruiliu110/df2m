import torch


def trace_fn(sigma_eps, temporal_cov, Kvv, Kuv, Kuu, beta_squared, jitter=1e-6):
    tmp1 = torch.sum(beta_squared)
    tmp2 = torch.trace(temporal_cov) - temporal_cov.shape[0]
    tmp3 = torch.trace(Kuu)
    dim_spatial = Kvv.shape[0]
    Kvv += jitter * torch.eye(dim_spatial)
    Kvv_tril = torch.linalg.cholesky(Kvv)
    spatial_tril_inverse = torch.linalg.solve_triangular(Kvv_tril, torch.eye(dim_spatial), upper=False)
    tmp4 = torch.sum((spatial_tril_inverse @ torch.transpose(Kuv, 0, 1)) ** 2.0)
    return 1. / 2. / (sigma_eps + jitter) * tmp1 * tmp2 * torch.maximum(tmp3 - tmp4, torch.Tensor([jitter]))
