from torch import nn as nn
import torch.nn.functional as F
import torch
import math
from .utils import BSFFL, TriSAGP ,SAGP
from .utils import wasserstein_distance, wasserstein_distance_matmul
from .embedding import LayerNorm

class MultiHeadedAttention(nn.Module):
    '''Fused Behavior-Aware Attention'''
    def __init__(self, h, n_b,  d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.n_b = n_b
        
        self.linear_layers_xm = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        self.linear_layers_xc = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        self.linear_layers_bm = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))
        self.linear_layers_bc = nn.Parameter(torch.randn(3, d_model, self.h, self.d_k))

        self.linear_layers_xm.data.normal_(mean=0.0, std=0.02)
        self.linear_layers_xc.data.normal_(mean=0.0, std=0.02)
        self.linear_layers_bm.data.normal_(mean=0.0, std=0.02)
        self.linear_layers_bc.data.normal_(mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

        self.mean_dense = nn.Linear(d_model, d_model)
        self.cov_dense = nn.Linear(d_model, d_model)
        self.out_dropout = nn.Dropout(dropout)

        self.LayerNorm = LayerNorm(d_model, eps=1e-12)

        self.Wq1 = nn.Linear(self.d_k, self.d_k)
        self.Wq2 = nn.Linear(self.d_k, self.d_k)
        self.Wk1 = nn.Linear(self.d_k, self.d_k)
        self.Wk2 = nn.Linear(self.d_k, self.d_k)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.h, self.d_k)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x_m, x_c , b_seq, b_m ,b_c, bb_m, bb_c, p_m, p_c, mask=None):
        batch_size, seq_len = x_m.size(0), x_m.size(1)

        queryx1, keyx1, valuex1 = x_m, x_m, x_m
        queryx2, keyx2, valuex2 = x_c, x_c, x_c
        queryb1, keyb1, valueb1 = b_m, b_m, b_m
        queryb2, keyb2, valueb2 = b_c, b_c, b_c

        #get behavior-specific queries, keys, and values
        queryx1, keyx1, valuex1= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_xm[l])
                        for l, x in zip(range(3), (queryx1, keyx1, valuex1))]
        queryx2, keyx2, valuex2= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_xc[l])
                        for l, x in zip(range(3), (queryx2, keyx2, valuex2))]
        queryb1, keyb1, valueb1= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_bm[l])
                        for l, x in zip(range(3), (queryb1, keyb1, valueb1))]
        queryb2, keyb2, valueb2= [torch.einsum("bnd, dhk->bhnk", x, self.linear_layers_bc[l])
                        for l, x in zip(range(3), (queryb2, keyb2, valueb2))]
        query1 = queryx1 + queryb1
        key1 = keyx1 + keyb1
        value1 = valuex1 + valueb1
        query2 = self.activation(queryx2 + queryb2) + 1
        key2 = self.activation(keyx2 + keyb2) + 1
        value2 = self.activation(valuex2 + valueb2) + 1

        # Position-enhanced behavior-aware fusion
        bb_m1 = self.transpose_for_scores(bb_m).permute(0, 3, 1, 2, 4).contiguous() # batch * h * b * b * k 
        bb_c1 = self.transpose_for_scores(bb_c).permute(0, 3, 1, 2, 4).contiguous()
        p_m1 = self.transpose_for_scores(p_m).permute(0, 2, 1, 3).contiguous()   # batch * h * n * k
        p_c1 = self.transpose_for_scores(p_c).permute(0, 2, 1, 3).contiguous()

        bb_m1_batch = torch.zeros(batch_size,self.h,seq_len,seq_len,self.d_k).cuda()
        bb_c1_batch = torch.zeros(batch_size,self.h,seq_len,seq_len,self.d_k).cuda()
        bb_m1_batch = bb_m1[torch.arange(batch_size)[:,None,None],:,b_seq[torch.arange(batch_size)][:,None],b_seq[torch.arange(batch_size)].unsqueeze(2),:].permute(0, 3, 2, 1, 4).contiguous()
        bb_c1_batch = bb_c1[torch.arange(batch_size)[:,None,None],:,b_seq[torch.arange(batch_size)][:,None],b_seq[torch.arange(batch_size)].unsqueeze(2),:].permute(0, 3, 2, 1, 4).contiguous()


        # query B * H * N * K -> B * H * N * j * K    (j:position)
        # fusionQ = B * H * N * N * K
        fusion_Q_m ,fusion_Q_c = TriSAGP(query1.unsqueeze(3), 
                                        self.Wq1(bb_m1_batch),
                                        self.Wq2(p_m1).unsqueeze(3), 
                                        query2.unsqueeze(3), 
                                        bb_c1_batch,
                                        p_c1.unsqueeze(3) )

        # key B * H * N * K -> B * H * i * N * K   (i:position)
        # fusionK = B * H * N * N * K   
        fusion_K_m ,fusion_K_c = TriSAGP(key1.unsqueeze(2), 
                                        self.Wk1(bb_m1_batch),
                                        self.Wk2(p_m1).unsqueeze(2), 
                                        key2.unsqueeze(2), 
                                        bb_c1_batch,
                                        p_c1.unsqueeze(2) )

        # wasserstein_distance attenation scores
        Wass_scores = -wasserstein_distance_matmul( fusion_Q_m.unsqueeze(4), fusion_Q_c.unsqueeze(4),
                                                    fusion_K_m.unsqueeze(4), fusion_K_c.unsqueeze(4)).squeeze()

        # Attentive aggregation. 
        Wass_scores = Wass_scores / math.sqrt(self.d_k)

        # dealing with padding and softmax.
        if mask is not None:
            assert len(mask.shape) == 2
            mask = (mask[:,:,None] & mask[:,None,:]).unsqueeze(1)
            if Wass_scores.dtype == torch.float16:
                Wass_scores = Wass_scores.masked_fill(mask == 0, -65500)
            else:
                Wass_scores = Wass_scores.masked_fill(mask == 0, -1e30)
        Wass_probs = self.dropout(nn.functional.softmax(Wass_scores, dim=-1))

        #attention_probs = self.attn_dropout(Wass_probs)

        mean_context_layer = torch.matmul(Wass_probs, value1)
        cov_context_layer = torch.matmul(Wass_probs ** 2, value2)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.h * self.d_k,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + x_m)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + x_c)

        return mean_hidden_states, cov_hidden_states, Wass_probs

class TransformerBlock(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, n_b, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        :param n_b: number of behaviors
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, n_b=n_b, d_model=hidden, dropout=dropout)
        self.feed_forward = BSFFL(d_model=hidden, d_ff=feed_forward_hidden, n_b=n_b, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.activation_func = nn.ELU()

    def forward(self, x_m, x_c, b_seq , b_m, b_c, bb_m, bb_c, p_m, p_c, mask):
        x_m, x_c, W_probs = self.attention(x_m, x_c , b_seq, b_m ,b_c, bb_m, bb_c, p_m, p_c,mask=mask)
        x_m = self.feed_forward(x_m,b_seq)
        x_c = self.activation_func(self.feed_forward(x_c,b_seq)) + 1
        return x_m, x_c, W_probs