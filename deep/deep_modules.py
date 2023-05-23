from torch import nn
import torch
import torch.optim as optim
from torch.nn.utils.parametrizations import spectral_norm
from matplotlib import pyplot as plt

from deep.attention import MaskedScaledDotProductAttention, MultiHeadMultiLayerAttention
from deep.linear_new import CustomLinear, CustomLayerNorm, CustomReluLinearLayer
from svgp import plot_loss


class CustomDL(nn.Module):
    def __init__(self, type_name, max_channels, feature_size, hidden_size, output_size, num_layers,
                 co_data=False, if_lip=True, dropout=0.0, number_heads=1):
        super(CustomDL, self).__init__()
        self.type_name = type_name
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_channels = max_channels
        self.feature_size = feature_size
        self.output_size = output_size
        self.dropout = dropout
        self.normalization = CustomLayerNorm(max_channels, feature_size,)
        self.in_layer = nn.Sequential(
            spectral_norm(CustomLinear(max_channels, feature_size, hidden_size)),
            nn.ReLU(),
            ) if if_lip else nn.Sequential(
            CustomLinear(max_channels, feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        if type_name == 'LSTM':
            self.lstm = spectral_norm(
                    spectral_norm(nn.LSTM(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True),
                                  'weight_hh_l0'),
                    'weight_ih_l0') if if_lip else nn.LSTM(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        elif type_name == 'GRU':
            self.gru = spectral_norm(
                spectral_norm(nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True),
                              'weight_hh_l0'),
                'weight_ih_l0') if if_lip else nn.GRU(hidden_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
        elif type_name == 'ATTENTION':
            self.attention = MultiHeadMultiLayerAttention(hidden_size, hidden_size, num_layers, num_heads=number_heads, attention_dropout=self.dropout, if_lip=if_lip)
        elif type_name == 'plain':
            pass
        elif type_name =='LINEAR':
            self.linear = CustomReluLinearLayer(hidden_size,hidden_size, num_layers, dropout_rate=self.dropout, if_lip=if_lip)

        self.fc = nn.Linear(hidden_size, output_size)

        self.has_initial_trained = True
        self.train_number = 0
        self.co_data = co_data

        self.initial_linear_layer = nn.Linear(self.output_size, self.feature_size * self.max_channels)

        if dropout is not None:
            self.dropout_layer = nn.Dropout(self.dropout)

    def forward_new(self, x, *args, **kwargs):
        if self.type_name == 'plain':
            return x[0].reshape(x.shape[1], -1)

        x = self.normalization(x)
        x = self.in_layer(x)

        if self.dropout is not None:
            x = self.dropout_layer(x)

        if self.type_name in ['LSTM', 'GRU']:
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size).requires_grad_()
            c0 = torch.zeros(self.num_layers, 1, self.hidden_size).requires_grad_()
            if self.type_name == 'LSTM':
                out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
            elif self.type_name == "GRU":
                out, hn = self.gru(x, h0.detach())
        elif self.type_name == 'ATTENTION':
            out = self.attention(x)
        elif self.type_name == 'LINEAR':
            out = self.linear(x)

        if self.dropout is not None:
            out = self.dropout_layer(out)
        out = self.fc(out)
        return out

    def forward(self, inputs, *args, **kwargs):
        inputs = torch.unsqueeze(inputs.permute(1,0,2), 0)
        results = self.forward_new(inputs, *args, **kwargs)
        return torch.squeeze(results, 0)

    def initial_train(self, inputs, initial_training_epochs=500,
                      initial_training_lr=0.001, loss_fn=nn.MSELoss(), *args, **kwargs,):
        optimizer = optim.Adam(self.parameters(), lr=initial_training_lr)
        inputs = inputs.permute(0,2,1,3)
        targets = inputs[:, 1:]
        loss_list = []
        for epoch in range(initial_training_epochs):
            self.zero_grad()
            outputs = self.initial_linear_layer(self.forward_new(inputs[:, :-1]))
            outputs = outputs.reshape(targets.shape)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        plot_loss(loss_list)
        plt.show()
        self.has_initial_trained = True






