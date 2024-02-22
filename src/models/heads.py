

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import wasserstein_distance,wasserstein_distance_matmul

# head used for bert4rec
class WassersteinPredictionHead(nn.Module):
    def __init__(self, d_model, num_items, token_embeddings_m, token_embeddings_c):
        super().__init__()
        self.token_embeddings_m = token_embeddings_m
        self.token_embeddings_c = token_embeddings_c
        self.vocab_size = num_items + 1
        self.out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ELU(),
            )
        self.activation = nn.ELU()
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x_m ,x_c , b_seq, candidates=None):
        x_m = self.out(x_m)  # B x H or M x H
        x_c = self.out(x_c)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb1 = self.token_embeddings_m(candidates)  # B x C x H
            emb2 = self.activation(self.token_embeddings_c(candidates)) + 1
            logits = wasserstein_distance_matmul(x_m.unsqueeze(1), x_c.unsqueeze(1), emb1, emb2).squeeze()

        else:  # x : M x H
            emb1 = self.token_embeddings_m.weight[:self.vocab_size]  # V x H
            emb2 = self.activation(self.token_embeddings_c.weight[:self.vocab_size]) + 1
            logits = wasserstein_distance_matmul(x_m.unsqueeze(1), x_c.unsqueeze(1), emb1, emb2).squeeze() # M x V

        return logits