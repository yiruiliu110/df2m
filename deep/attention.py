import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class MaskedScaledDotProductAttention(nn.Module):
    def __init__(self, input_size, hidden_size,  attention_dropout=None, if_lip=True):


        super(MaskedScaledDotProductAttention, self).__init__()
        self.attention_dropout = attention_dropout
        if attention_dropout is not None:
            self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.q_linear = spectral_norm(nn.Linear(input_size, hidden_size)) if if_lip else nn.Linear(input_size, hidden_size)
        self.k_linear = spectral_norm(nn.Linear(input_size, hidden_size)) if if_lip else nn.Linear(input_size, hidden_size)
        self.v_linear = spectral_norm(nn.Linear(input_size, hidden_size)) if if_lip else nn.Linear(input_size, hidden_size)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)

        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)

        # Create mask to prevent using future information
        seq_len = q.size(1)
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).to(q.device)
        scores = scores.masked_fill(mask == 1, float('-inf'))

        attention = self.softmax(scores)
        if self.attention_dropout is not None:
            attention = self.dropout(attention)
        return torch.matmul(attention, v), attention


class MultiHeadMultiLayerAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads, attention_dropout=None, if_lip=True):
        super(MultiHeadMultiLayerAttention, self).__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size

        self.layers = nn.ModuleList([nn.ModuleList([MaskedScaledDotProductAttention(input_size, hidden_size, attention_dropout, if_lip)
                                                    for _ in range(num_heads)])
                                     for _ in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        for layer in self.layers:
            attention_list = []

            for head in layer:
                attention_head, _ = head(x)
                attention_list.append(attention_head)

            # Concatenate attention heads
            concat_attention = torch.cat(attention_list, dim=2)

            # Pass through the final linear layer
            output = nn.Linear(self.hidden_size * len(layer), self.hidden_size)(concat_attention)

            x = output

        return x
