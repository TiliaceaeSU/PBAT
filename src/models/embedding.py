
from torch import nn as nn
import torch

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class PositionalEmbedding(nn.Module):
    
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.apply(self._init_weights)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class SimpleEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        self.LayerNorm = LayerNorm(embed_size, eps=1e-12)
        self.embed_size = embed_size
        self.activation = nn.ELU()
        self.apply(self._init_weights)

    def forward(self, sequence):
        x = self.token(sequence)
        x = self.LayerNorm(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()