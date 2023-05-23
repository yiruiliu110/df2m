import numpy as np
from matplotlib import pyplot as plt


def np_z_trim(x, threshold=10, axis=0):
    """ Replace outliers in numpy ndarray along axis with min or max values
    within the threshold along this axis, whichever is closer."""
    x = np.array(x)
    mean = np.median(x, axis=axis, keepdims=True)
    std = np.quantile(x, 0.6333,  axis=axis, keepdims=True) - np.quantile(x, 0.5,  axis=axis, keepdims=True)
    masked = np.where(np.abs(x - mean) < threshold * std, x, np.nan)
    min = np.nanmin(masked, axis=axis, keepdims=True)
    max = np.nanmax(masked, axis=axis, keepdims=True)
    repl = np.where(np.abs(x - max) < np.abs(x - min), max, min)
    return np.where(np.isnan(masked), repl, masked)


# visualize the result
def print_plots(index, dim_index, train_x, train_y, mean, lower, upper):
    fig, (func, samp) = plt.subplots(1, 2, figsize=(12, 3))
    line, = func.plot(train_x.numpy(), mean[index, dim_index, :].detach().cpu().numpy(), label='GP prediction')
    func.fill_between(
        train_x.numpy(), lower[index, dim_index, :].detach().cpu().numpy(),
        upper[index, dim_index, :].detach().cpu().numpy(), color=line.get_color(), alpha=0.5
    )
    # sample from p(y|D,x) = \int p(y|f) p(f|D,x) df (doubly stochastic)
    samp.scatter(train_x.numpy(), train_y[index, dim_index, :].squeeze().numpy(), alpha=0.5, label='True train data')

    samp.legend()
    plt.show()


# Here's a quick helper function for getting smoothed percentile values from samples
def percentiles_from_samples(samples, percentiles=[0.05, 0.5, 0.95]):
    num_samples = samples.size(0)
    samples = samples.sort(dim=0)[0]

    # Get samples corresponding to percentile
    percentile_samples = [samples[int(num_samples * percentile)] for percentile in percentiles]

    return percentile_samples