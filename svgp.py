import os
import matplotlib.pyplot as plt
import torch
import numpy as np

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

smoke_test = "CI" in os.environ
assert pyro.__version__.startswith('1.8.4')
pyro.set_rng_seed(0)


def plot(
        X_inputs, y_inputs,
        y_index=None,
        plot_observed_data=False,
        plot_predictions=False,
        n_prior_samples=0,
        model=None,
        kernel=None,
        n_test=500,
        ax=None,
):
    if y_index is not None:
        y_inputs = y_inputs[y_index]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    if plot_observed_data:
        ax.plot(X_inputs.numpy(), y_inputs.numpy(), "kx")
    if plot_predictions:
        Xtest = X_inputs  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            mean, cov = model(Xtest, full_cov=False)
            if y_index is not None:
                mean = mean[y_index]
                cov = cov[y_index]

        sd = cov**0.5  # standard deviation at each input point x
        ax.plot(Xtest.numpy(), mean.numpy(), "r", lw=2)  # plot the mean
        ax.fill_between(
            Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
            (mean - 2.0 * sd).numpy(),
            (mean + 2.0 * sd).numpy(),
            color="C0",
            alpha=0.3,
        )
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(-0.5, 5.5, n_test)  # test inputs
        noise = (
            model.noise
            if type(model) != gp.models.VariationalSparseGP
            else model.likelihood.variance
        )
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(
            torch.zeros(n_test), covariance_matrix=cov
        ).sample(sample_shape=(n_prior_samples,))
        ax.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    #ax.set_xlim(-0.5, 5.5)


def plot_inducing_points(Xu, ax=None):
    for xu in Xu:
        g = ax.axvline(xu, color="red", linestyle="-.", alpha=0.5)
    ax.legend(
        handles=[g],
        labels=["Inducing Point Locations"],
        bbox_to_anchor=(0.5, 1.15),
        loc="upper center",
    )


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("Iterations")
    _ = plt.ylabel("Loss")  # supress output text


def main():
    N = 1000
    X = dist.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    y = torch.cat(
        [torch.cat([0.5 * torch.sin(3 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(4 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(5 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(6 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N))], 0).unsqueeze(0),
         torch.cat([0.5 * torch.sin(7 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(8 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(9 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N)),
                    0.5 * torch.sin(10 * X) + dist.Normal(0.0, 0.2).sample(sample_shape=(1, N))], 0).unsqueeze(0)],
        0)

    Xu = torch.arange(20.0) / 4.0

    for i in range(2):
        for j in range(4):
            plot(X, y, y_index=(i, j), plot_observed_data=True)
            # initialize the inducing inputs
            plot_inducing_points(Xu, plt.gca())
            plt.show()

    # initialize the kernel and model
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim=1)
    likelihood = gp.likelihoods.Gaussian()
    # we increase the jitter for better numerical stability
    vsgp = gp.models.VariationalSparseGP(
        X, y, kernel, Xu=Xu, likelihood=likelihood, jitter=1.0e-5, whiten=True
    )

    # the way we setup inference is similar to above
    optimizer = torch.optim.Adam(vsgp.parameters(), lr=0.005)
    loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
    losses = []
    locations = []
    variances = []
    lengthscales = []
    # noises = []
    num_steps = 2000 if not smoke_test else 2
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = loss_fn(vsgp.model, vsgp.guide)
        locations.append(vsgp.Xu.data.numpy().copy())
        variances.append(vsgp.kernel.variance.item())
        # noises.append(vsgp.noise.item())
        lengthscales.append(vsgp.kernel.lengthscale.item())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    plot_loss(losses)
    plt.show()

    for i in range(2):
        for j in range(4):
            plot(X_inputs=X, y_inputs=y, y_index=(i, j), model=vsgp, plot_observed_data=True, plot_predictions=True)
            plot_inducing_points(vsgp.Xu.data.numpy(), plt.gca())
            plt.show()


if __name__ == "__main__":
    main()
