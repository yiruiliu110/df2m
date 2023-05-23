import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class CustomLayerNorm(nn.Module):
    def __init__(self, max_channels, feature_size, eps=1e-5):
        super(CustomLayerNorm, self).__init__()
        self.max_channels = max_channels
        self.feature_size = feature_size
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(max_channels * feature_size))
        self.beta = nn.Parameter(torch.zeros(max_channels * feature_size))

    def forward(self, x):
        batch_size, timesteps, channels, feature_size = x.shape
        assert channels <= self.max_channels, "Input has more channels than expected"
        x = x.reshape(batch_size, timesteps, -1)

        gamma = self.gamma[: channels * feature_size]
        beta = self.beta[: channels * feature_size]

        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / (std + self.eps)
        x_scaled_shifted = x_normalized * gamma + beta

        x_scaled_shifted  = x_scaled_shifted.reshape(batch_size, timesteps, channels, feature_size)

        return x_scaled_shifted



class CustomLinear(nn.Module):
    def __init__(self, max_channels, feature_size, output_size):
        super(CustomLinear, self).__init__()
        self.max_channels = max_channels
        self.feature_size = feature_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.randn(max_channels * feature_size, output_size))
        self.bias = nn.Parameter(torch.randn(output_size))


    def forward(self, x):
        batch_size, timesteps, channels, feature_size = x.shape
        assert channels <= self.max_channels, "Input has more channels than expected"
        x = x.reshape(batch_size, timesteps, -1)

        weight = self.weight[: channels * feature_size, :]
        x = torch.matmul(x, weight) + self.bias

        return x


class CustomReluLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, num_layers, dropout_rate=0.5, if_lip=False):
        super(CustomReluLinearLayer, self).__init__()
        self.if_lip = if_lip
        self.num_layers = num_layers
        self.linears = nn.ModuleList([spectral_norm(nn.Linear(in_features if i == 0 else out_features, out_features)) if if_lip
                                      else nn.Linear(in_features if i == 0 else out_features, out_features)
                                      for i in range(num_layers)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_layers)])
        self.relus = nn.ModuleList([nn.ReLU() for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.linears[i](x)
            x = self.dropouts[i](x)
            x = self.relus[i](x)
        return x
