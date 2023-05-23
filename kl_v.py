import torch


def kl_v(v, s_tril, temporal_cov, spatial_cov, jitter=1e-6):
    dim_spatial = spatial_cov.shape[0]
    dim_temporal = temporal_cov.shape[0]
    old_jitter = jitter + 0.0
    flag = True
    i = 0
    while flag:
        try:
            temporal_cov += jitter * torch.eye(dim_temporal)
            temporal_tril = torch.linalg.cholesky(temporal_cov)
            flag = False
        except:
            i += 1
            print('temporal noninversible!')
            jitter *= 10
            if i == 4:
                temporal_tril = torch.eye(temporal_cov.shape[0])
                flag = False

    jitter = old_jitter + 0.0
    flag = True
    while flag:
        try:
            spatial_cov += jitter * torch.eye(dim_spatial)
            spatial_tril = torch.linalg.cholesky(spatial_cov)
            flag = False
        except:
            print('spatial noninversible!')
            jitter *= 10

    temporal_tril_inverse = torch.linalg.solve_triangular(temporal_tril, torch.eye(dim_temporal), upper=False)
    spatial_tril_inverse = torch.linalg.solve_triangular(spatial_tril, torch.eye(dim_spatial), upper=False)

    temporal_tril_inverse_diag = torch.diagonal(temporal_tril_inverse)
    spatial_tril_inverse_transpose = torch.transpose(spatial_tril_inverse, 1, 0)

    tmp2 = torch.matmul(torch.matmul(temporal_tril_inverse, v), spatial_tril_inverse_transpose)
    add2 = torch.sum(tmp2 ** 2.0)

    tmp3 = torch.matmul(torch.matmul(torch.eye(dim_temporal), v), spatial_tril_inverse_transpose)
    add3 = torch.sum(tmp3 ** 2.0)

    tmp1 = torch.sum(torch.matmul(spatial_tril_inverse, s_tril) ** 2, dim=[2,3])
    add1 = torch.einsum('k, bk -> b', torch.sum(temporal_tril_inverse ** 2.0, dim=1) - 1.0, tmp1)

    add4 = 2 * torch.sum(torch.log(torch.abs(temporal_tril_inverse_diag)+1e-15))

    return (torch.sum(add1) + torch.sum(add2) - torch.sum(add3) * 1.0 - dim_spatial * torch.sum(add4))/2.0
