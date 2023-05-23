import pyro
import numpy as np
import skfda
from pyro.contrib.gp.models import VariationalSparseGP
from pyro.distributions import Bernoulli
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA
from deep.deep_modules import CustomDL
from kl_v import kl_v
from kl_z_and_a import model_ibp, model_a, guide_ibp, guide_a
import torch
from pyro.nn.module import pyro_method
from pyro.contrib.gp.parameterized import Parameterized
import pyro.poutine as poutine
import pyro.contrib.gp as gp

from trace_fn import trace_fn


class DF2M(Parameterized):
    def __init__(self,
                 X,
                 y,
                 kernel,
                 Xu,
                 num_clusters=5,
                 num_dim=40,
                 num_inducing=40,
                 whiten=False,
                 jitter=1e-6,
                 dl_module_type='LSTM',
                 if_positive=True,
                 initial_alpha=10.,
                 hidden_size=5,
                 out_size=5,
                 co_data=True
                 ):
        super().__init__()
        self.hidden_size = hidden_size
        self.out_size = out_size
        if dl_module_type == 'plain':
            self.out_size = num_inducing * num_clusters
        self.co_data = co_data
        self.y = y
        self.X = X
        self.num_periods = self.y.shape[1]
        self.num_points = self.y.shape[-1]
        latent_shape = torch.Size([num_clusters, self.num_periods])

        self.likelihood = gp.likelihoods.Gaussian(torch.Tensor([0.1]))

        self.vsgp = VariationalSparseGP(X, y=None, kernel=kernel, Xu=Xu, likelihood=None,
                                        latent_shape=latent_shape, whiten=whiten, jitter=jitter)
        self.jitter = jitter
        self.num_clusters = num_clusters
        self.num_dim = num_dim
        self.num_inducing = num_inducing

        self.dl_module_type = dl_module_type
        self.set_up_deep_kernel()

        self.if_positive = if_positive

        self.initial_alpha = initial_alpha
        self.standard_normal = pyro.distributions.Normal(0,1)


    @pyro_method
    def guide(self):
        self.set_mode("guide")
        self.z_logits, self.post_z = guide_ibp(self.num_clusters, self.num_dim, jitter=self.jitter, initial_alpha=self.initial_alpha)
        if not self.if_positive:
            self.mat_a_mean, self.mat_a_std = guide_a(self.num_clusters, self.num_dim)
        else:
            self.mat_a_concntration, self.mat_a_rate = guide_a(self.num_clusters, self.num_dim, if_positive=self.if_positive)
        self.vsgp.guide()


    @pyro_method
    def model(self):
        self.set_mode("model")
        f_loc, f_var = self.vsgp.model()
        self.alpha, z = model_ibp(self.num_clusters, self.num_dim, jitter=self.jitter, initial_alpha=self.initial_alpha)
        mat_a = model_a(self.num_clusters, self.num_dim, if_positive=self.if_positive)
        beta = z * mat_a

        y_loc = torch.einsum('...ij, ...jkl-> ...ikl', beta, f_loc)
        y_var = torch.einsum('...ij, ...jkl-> ...ikl', beta * beta, f_var)

        self.likelihood._load_pyro_samples()
        with poutine.scale(scale=torch.Tensor([1.0])):
            return self.likelihood(y_loc, y_var, self.y)

    def average_beta(self):
        mean_z = torch.sigmoid(self.z_logits)
        if not self.if_positive:
            mean_a = self.mat_a_mean
        else:
            mean_a = self.mat_a_concntration / self.mat_a_rate
        beta_average = mean_z * mean_a
        return beta_average

    def second_moment_beta(self):
        mean_z = torch.sigmoid(self.z_logits)
        if not self.if_positive:
            mean_a, std_a = self.mat_a_mean, self.mat_a_std
            return mean_z * (mean_a ** 2.0 + std_a ** 2.0)
        else:
            var_a = self.mat_a_concntration / (self.mat_a_rate ** 2.0)
            return mean_z * var_a

    def forward(self, Xnew, full_cov=False):
        self.set_mode("guide")
        x_loc, x_var = self.vsgp.forward(Xnew, full_cov)

        beta_average = self.average_beta()

        y_loc = torch.einsum('ij, jkl-> ikl', beta_average, x_loc)
        y_var = torch.einsum('ij, jkl-> ikl', beta_average * beta_average, x_var)
        return y_loc, y_var

    def set_up_deep_kernel(self):
        self.dl_module = CustomDL(type_name=self.dl_module_type,
                                  max_channels=self.num_dim if self.co_data else self.num_clusters,
                                  feature_size=self.num_points if self.co_data else self.num_inducing,
                                  hidden_size=self.hidden_size,
                                  output_size=self.out_size,
                                  num_layers=1,
                                  if_lip=True,
                                  co_data=self.co_data)

        rbf = gp.kernels.RationalQuadratic(input_dim=self.out_size)
        constant = gp.kernels.Constant(input_dim=self.out_size)
        inner_kernel = gp.kernels.Sum(rbf, constant)
        self.temporal_kernel = gp.kernels.Warping(inner_kernel, iwarping_fn=self.dl_module)

    def value_temporal_kernel(self, batch_size=1, add_v=None):
        v = self.get_v_sample_batch(batch_size)
        if add_v is not None:
            v = torch.cat([v, add_v], dim=1 if batch_size==1 else 2)
        temporal_cov = torch.stack([self.temporal_kernel(v[i]) for i in range(batch_size)])
        temporal_cov = torch.mean(temporal_cov, dim=0)
        return temporal_cov

    def get_v(self):
        v = self.vsgp.u_loc
        return v

    def get_v_sample_batch(self, batch_size=1):
        v = self.get_v()
        v = self.factor_select(v)
        s_tril = self.get_s()
        s_tril = self.factor_select(s_tril)
        if batch_size==1:
            samples = self.standard_normal.sample(v.shape)
            results = v + torch.einsum('bkjl, bkl-> bkj', s_tril, samples)
        else:
            samples = self.standard_normal.sample([batch_size, self.active_number,
                                                   self.num_periods, self.num_inducing])
            results = v + torch.einsum('bkjl, sbkl-> sbkj', s_tril, samples)
        return results


    def get_s(self):
        s_tril = self.vsgp.u_scale_tril
        return s_tril

    def get_spatial_cov_vv(self):
        Kvv = self.vsgp.kernel(self.vsgp.Xu).contiguous()
        return Kvv

    def get_spatial_cov_uv(self):
        Kuv = self.vsgp.kernel(self.vsgp.X, self.vsgp.Xu).contiguous()
        return Kuv

    def get_spatial_cov_uu(self):
        Kuu = self.vsgp.kernel(self.vsgp.X, self.vsgp.X).contiguous()
        return Kuu

    def compute_temporal_cov(self, no_gradient, batch_size=1):
        if not self.co_data:
            if no_gradient:
                with torch.no_grad():
                    temporal_cov = self.value_temporal_kernel(batch_size)
            else:
                temporal_cov = self.value_temporal_kernel(batch_size)
            return temporal_cov
        else:
            return self.temporal_kernel(self.y)

    def z_gate(self):
        self.active_index = torch.nonzero(torch.sum(self.post_z, dim=0))
        self.active_number = self.active_index.shape[0]

    def factor_select(self, inputs):
        return inputs[:self.active_number, ...]

    def temporal_kernel_fn(self, no_gradient=False, batch_size=1):
        self.z_gate()
        temporal_cov = self.compute_temporal_cov(no_gradient, batch_size)[:-1, :-1]
        spatial_cov = self.get_spatial_cov_vv()
        v = self.get_v()
        v = self.factor_select(v)
        s = self.get_s()
        s = self.factor_select(s)
        if no_gradient:
            return kl_v(v[:, 1:, :], s[:, 1:, :], temporal_cov.detach(), spatial_cov)
        else:
            return kl_v(v[:, 1:, :].detach(), s[:, 1:, :].detach(), temporal_cov, spatial_cov.detach())

    def adjust_trace_fn(self, no_gradient=False, batch_size=1):
        temporal_cov = self.compute_temporal_cov(no_gradient, batch_size)[:-1, :-1]
        Kvv = self.get_spatial_cov_vv()
        Kuv = self.get_spatial_cov_uv()
        Kuu = self.get_spatial_cov_uu()
        epsilon = self.likelihood.variance
        beta_squared = self.second_moment_beta()
        if no_gradient:
            return trace_fn(epsilon, temporal_cov.detach(), Kvv, Kuv, Kuu, beta_squared, self.jitter)
        else:
            return trace_fn(epsilon.detach(), temporal_cov, Kvv.detach(), Kuv.detach(), Kuu.detach(), beta_squared.detach(), self.jitter)

    def prediction(self, h_max=3, batch_size=50, sample_number=50):
        y_predict_record = [[], [], []]
        v_predict_record = [[], [], []]
        for _ in range(sample_number):
            inverse_kvv = torch.linalg.pinv(self.get_spatial_cov_vv())
            kuv = self.get_spatial_cov_uv()
            beta = self.average_beta()
            beta = beta[:, :self.active_number]
            y_predict = []
            v_predict = []

            temporal_cov = self.value_temporal_kernel(batch_size=batch_size)
            v = self.get_v()
            v = self.factor_select(v)
            inputs = self.factor_select(v)

            for i in range(h_max):
                kxx = temporal_cov[:-1, :-1]
                inverse_kxx = torch.linalg.pinv(kxx)
                kx1 = temporal_cov[:-1, -1:]
                v_next_period = torch.einsum('bjk, kl -> bjl', v.permute([0, 2, 1])[:, :, 1:], inverse_kxx @ kx1)
                x_next_period = torch.einsum('jk, bkl -> bjl',
                                             kuv @ inverse_kvv,
                                             v_next_period
                                             ).permute([0, 2, 1])
                y_next_period = torch.einsum('jk, klr -> jlr',
                                             beta,
                                             x_next_period)
                y_predict.append(y_next_period)
                self.y = torch.cat([self.y, y_next_period], dim=1)
                v_next_period = v_next_period.permute([0, 2, 1])
                v_predict.append(v_next_period)

                v = torch.cat([v, v_next_period], dim=1)
                inputs = torch.cat([inputs, self.factor_select(v_next_period)], dim=1)
                temporal_cov = self.temporal_kernel(self.y)if self.co_data else self.temporal_kernel(inputs)

                y_predict_record[i].append(torch.unsqueeze(y_next_period, 0))
                v_predict_record[i].append(torch.unsqueeze(v_next_period, 0))

        y_predict_output = [torch.mean(torch.cat(item, dim=0), dim=0) for item in y_predict_record]
        v_predict_output = [torch.mean(torch.cat(item, dim=0), dim=0) for item in y_predict_record]
        return y_predict_output, v_predict_output

    def set_initial_factors(self, n_basis=20):
        n_components = self.num_clusters
        n_basis = max(20, n_components)
        with torch.no_grad():
            data_matrix = self.y.reshape(-1, self.num_points)
            grid_points = self.X

            fd = FDataGrid(data_matrix.tolist(), grid_points.tolist())
            basis = skfda.representation.basis.BSplineBasis(n_basis=n_basis)
            basis_fd = fd.to_basis(basis)
            fpca_basis = FPCA(n_components=n_components)
            fpca_basis = fpca_basis.fit(basis_fd)

            pca_components = fpca_basis.components_
            for i in range(self.num_clusters):
                if i < n_components:
                    y_value = pca_components.evaluate(self.vsgp.Xu.numpy())
                    y_value = np.squeeze(y_value)
                    self.vsgp.u_loc.data[i, :] = torch.Tensor(y_value[i])
                else:
                    self.vsgp.u_loc.data[i, :] = torch.Tensor(y_value[0]) * 0.0







