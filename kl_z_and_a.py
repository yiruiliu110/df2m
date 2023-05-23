import pyro
import torch
import pyro.distributions as dist
from pyro.distributions import constraints

from pyro.distributions import Bernoulli, Normal

def model_ibp(num_clusters, num_dim, jitter=1e-6, initial_alpha=10.):
    alpha = pyro.param('alpha', initial_alpha * torch.Tensor([1.0]), constraint=constraints.positive)
    prop = pyro.sample('prop',
                       dist.Beta(concentration1=torch.ones(num_clusters) * alpha,
                                 concentration0=torch.ones(num_clusters)).to_event())
    cum_prop = torch.cumprod(prop, dim=0)
    cum_prop_logit = torch.logit(jitter + cum_prop * (1 - jitter))
    tmp_dist = Bernoulli(logits=torch.unsqueeze(cum_prop_logit, -2)).expand(
        [num_dim, num_clusters]).to_event(2)
    z = pyro.sample('z', tmp_dist)
    return alpha, z


def guide_ibp(num_clusters, num_dim, jitter, initial_alpha=10.):
    prop_alpha = pyro.param("prop_alpha", initial_alpha * torch.ones(num_clusters), constraint=constraints.positive)
    prop_beta = pyro.param("prop_beta", torch.ones(num_clusters), constraint=constraints.positive)

    prop = pyro.sample('prop', dist.Beta(concentration1=prop_alpha, concentration0=prop_beta).to_event())
    cum_prop = torch.cumprod(prop, dim=0)
    #z_logits = torch.logit(jitter + cum_prop * (1 - jitter))
    #tmp_dist = Bernoulli(logits=torch.unsqueeze(z_logits, -2)).expand(
    #    [num_dim, num_clusters]).to_event(2)
    #z = pyro.sample('z', tmp_dist)
    z_logits = pyro.param("z_logits", torch.zeros(num_dim, num_clusters))
    z = pyro.sample('z', Bernoulli(logits=z_logits).to_event(2))
    return z_logits, z


def model_a(num_clusters, num_dim, if_positive=False):
    if not if_positive:
        sigma_A = pyro.param('sigma_A_prior', torch.ones(1), constraint=constraints.positive)
        mat_a = pyro.sample('mat_A',
                            dist.Normal(torch.zeros(num_dim, num_clusters),
                                        torch.ones(num_dim, num_clusters)*sigma_A).to_event())
    else:
        mat_a_concentration = pyro.param('mat_a_concentration_prior', torch.ones(1), constraint=constraints.positive)
        mat_a_rate = pyro.param('mat_a_rate_prior', torch.ones(1), constraint=constraints.positive)
        mat_a = pyro.sample('mat_A',
                            dist.Gamma(torch.ones(num_dim, num_clusters) * mat_a_concentration,
                                       torch.ones(num_dim, num_clusters) * mat_a_rate).to_event())
    return mat_a


def guide_a(num_clusters, num_dim, if_positive=False):
    if not if_positive:
        mat_a_mean = pyro.param("mat_a_mean", torch.zeros(num_dim, num_clusters))
        mat_a_std = pyro.param("mat_a_std", torch.ones(num_dim, num_clusters),
                               constraint=constraints.positive)
        mat_a = pyro.sample('mat_A',
                            dist.Normal(mat_a_mean, mat_a_std).to_event())
        return mat_a_mean, mat_a_std
    else:
        mat_a_concentration = pyro.param('mat_a_concentration', torch.ones(num_dim, num_clusters), constraint=constraints.positive)
        mat_a_rate = pyro.param('mat_a_rate', torch.ones(num_dim, num_clusters), constraint=constraints.positive)
        mat_a = pyro.sample('mat_A',
                            dist.Gamma(mat_a_concentration, mat_a_rate).to_event())
        return mat_a_concentration, mat_a_rate

