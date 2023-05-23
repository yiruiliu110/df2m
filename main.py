import torch
from torch import nn
from data.data_processing import standardize
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.poutine as poutine
from df2m import DF2M
from data.read_data import read_data
from tqdm import trange


def run_one_period(
        data_name,
        deep_learning_module_name,
        num_inducing=30,
        num_clusters=5,
        num_particles=10,  # for elbo
        small_lr=0.001,
        big_lr=0.05,
        num_steps=500,
        if_positive=True,
        rolling_training_index=0,
        initial_alpha=10.,
        hidden_size=5,
        batch_size=10,
        max_value=1.0,
        if_co_data=True,
):
    data_np, num_dim, num_tasks, num_points = read_data(data_name)

    data_non_stand = torch.from_numpy(data_np)
    data, means, std = standardize(data_non_stand)

    train_x = torch.linspace(0., num_points - 1, num_points)
    train_y = data[:, rolling_training_index:num_tasks + rolling_training_index, :]  # torch.Size([47, 33, 96])

    pyro.clear_param_store()
    kernel = gp.kernels.RationalQuadratic(1)
    Xu = torch.cat([torch.linspace(1.0, num_points - 1., num_inducing)], 0)
    # we increase the jitter for better numerical stability

    df2m = DF2M(
        X=train_x, y=train_y, kernel=kernel, Xu=Xu, num_clusters=num_clusters,
        num_dim=num_dim,
        num_inducing=num_inducing, whiten=False,
        dl_module_type=deep_learning_module_name,
        if_positive=if_positive,
        initial_alpha=initial_alpha,
        hidden_size=hidden_size,
        co_data=if_co_data
    )

    loss_fn_0 = pyro.infer.Trace_ELBO(num_particles=num_particles).differentiable_loss
    loss_fn = pyro.infer.Trace_ELBO(num_particles=num_particles).differentiable_loss
    # the way we set up inference is similar to above
    with poutine.trace(param_only=True) as param_capture:
        loss = loss_fn(df2m.model, df2m.guide)
    params = set(
        site["value"].unconstrained() for site in param_capture.trace.nodes.values()
    )
    optimizer = torch.optim.Adam(params, lr=big_lr)
    optimizer_2 = torch.optim.Adam(params, lr=small_lr)

    with poutine.trace(param_only=True) as param_capture_1:
        loss_1 = df2m.temporal_kernel_fn(batch_size=batch_size) + df2m.adjust_trace_fn(batch_size=batch_size)
    params_1 = set(
        site["value"].unconstrained() for site in param_capture_1.trace.nodes.values()
    )

    optimizer_1 = torch.optim.Adam(params_1, lr=small_lr)

    losses = []
    losses_1 = []
    print('star training')

    for _ in trange(num_steps):
        optimizer.zero_grad()
        loss = loss_fn_0(df2m.model, df2m.guide)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    for _ in trange(num_steps):
        optimizer_1.zero_grad()
        loss_1 = df2m.temporal_kernel_fn(batch_size=batch_size)
        loss_1.backward()
        optimizer_1.step()
        losses_1.append(loss_1.item())

    losses = [item1 + item2 for item1, item2 in zip(losses, losses_1)]

    for _ in trange(num_steps * 2):
        optimizer_2.zero_grad()
        loss_old = loss_fn(df2m.model, df2m.guide)
        loss_new = df2m.temporal_kernel_fn(no_gradient=True, batch_size=batch_size) + df2m.adjust_trace_fn(
            no_gradient=True, batch_size=batch_size)
        loss = loss_old + loss_new
        loss.backward()
        torch.nn.utils.clip_grad_value_(params, max_value)
        optimizer_2.step()

        optimizer_1.zero_grad()
        loss_11 = df2m.temporal_kernel_fn(batch_size=batch_size)
        loss_12 = df2m.adjust_trace_fn(batch_size=batch_size)
        loss_1 = loss_11 + loss_12
        loss_1.backward()
        torch.nn.utils.clip_grad_value_(params_1, max_value)
        optimizer_1.step()

        losses_1.append(loss_1.item())
        losses.append(loss.item())


    ###  prediction
    y_predict, v_predict = df2m.prediction()

    h_max = 3
    test_data = data[:, num_tasks + rolling_training_index:, :]

    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()
    mae = []
    mse = []
    for i in range(h_max):
        pred = y_predict[i]
        true = test_data[:, i:i + 1]
        pred *= std
        true *= std
        mae.append(float(mae_loss(pred, true).detach().numpy()))
        mse.append(float(mse_loss(pred, true).detach().numpy()))

    print(data_name, deep_learning_module_name)
    print('time index', rolling_training_index, 'if_positive', if_positive)
    print('mae', mae)
    print('mse', mse)

    return mae, mse


if __name__ == "__main__":
    run_one_period('jp', 'ATTENTION',
                   rolling_training_index=0,
                   if_positive=False,
                   big_lr=0.05,
                   small_lr=0.001,
                   num_steps=500,
                   num_inducing=20,
                   num_clusters=10,
                   hidden_size=16,
                   initial_alpha=5.,
                   if_co_data=False)
