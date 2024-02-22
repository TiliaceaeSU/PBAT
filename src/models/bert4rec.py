
import torch
from torch import nn as nn
import pytorch_lightning as pl
from .embedding import SimpleEmbedding,PositionalEmbedding
from .new_transformer import TransformerBlock
from .utils import SAGP, wasserstein_distance_matmul


class BERT(pl.LightningModule):
    def __init__(self,
        max_len: int = None,
        num_items: int = None,
        n_layer: int = None,
        n_head: int = None,
        num_users: int = None,
        n_b: int = None,
        d_model: int = None,
        dropout: float = .0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_items = num_items
        self.num_users = num_users
        self.max_len = max_len
        self.n_b = n_b
        self.activation = nn.ELU()

        # SAGP
        # self.Wub = nn.Parameter(torch.randn(self.batch_size,self.n_b+1,self.d_model))
        self.Wub = nn.Linear(d_model, d_model)
        # self.WPub = nn.Parameter(torch.randn(self.batch_size,self.max_len,self.d_model))
        self.WPub = nn.Linear(d_model, d_model)

        #vocab_size = num_items + 1 + n_b # add padding and mask 


        ''' Dynamic Representation Encoding'''
        # entity distributions
        self.item_embedding_m = SimpleEmbedding(vocab_size=num_items+2, embed_size=d_model, dropout=dropout)
        self.item_embedding_c = SimpleEmbedding(vocab_size=num_items+2, embed_size=d_model, dropout=dropout)
        self.b_embedding_m = SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout)
        self.b_embedding_c = SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout)
        self.u_embedding_m = SimpleEmbedding(vocab_size=num_users+1, embed_size=d_model, dropout=dropout)
        self.u_embedding_c = SimpleEmbedding(vocab_size=num_users+1, embed_size=d_model, dropout=dropout)
        self.p_embedding_m = PositionalEmbedding(max_len=max_len, d_model=d_model)
        self.p_embedding_c = PositionalEmbedding(max_len=max_len, d_model=d_model)

        # behavior-relation distributions
        #self.bb_embedding_m = nn.ModuleList([SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout) for _ in range(self.n_b+1)])
        #self.bb_embedding_c = nn.ModuleList([SimpleEmbedding(vocab_size=n_b+1, embed_size=d_model, dropout=dropout) for _ in range(self.n_b+1)])
        self.bb_embedding_m = SimpleEmbedding(vocab_size=n_b*n_b+1, embed_size=d_model, dropout=dropout)
        self.bb_embedding_c = SimpleEmbedding(vocab_size=n_b*n_b+1, embed_size=d_model, dropout=dropout)
        
        # multi-layers transformer blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(d_model, n_head, d_model * 4, n_b, dropout) for _ in range(n_layer)])

    def forward(self, x, b_seq, u_id):

        mask = (x > 0)
        x_m = self.item_embedding_m(x)
        x_c = self.item_embedding_c(x) + 1
        b_m = self.b_embedding_m(b_seq)
        b_c = self.b_embedding_c(b_seq)+1
        u_m = self.u_embedding_m(u_id).squeeze()
        u_c = self.u_embedding_c(u_id).squeeze()+1
        p_m = self.p_embedding_m(x)
        p_c = self.p_embedding_c(x)+1

        batch_size = u_id.size(0)

        '''Personalized Pattern Learning'''

        b_mi = self.b_embedding_m(torch.LongTensor(u_id.size(0)*[[0,1,2,3,4]]).cuda()).squeeze()
        b_ci = self.b_embedding_c(torch.LongTensor(u_id.size(0)*[[0,1,2,3,4]]).cuda()).squeeze()
        b_ci = self.activation(b_ci) + 1
        
        # Personalized patterns distributions
        # B * n_b *H
        P_user_behavior_m , P_user_behavior_c = SAGP(u_m.unsqueeze(1),self.Wub(b_mi),u_c.unsqueeze(1),b_ci)

        ''' Behavioral Collaboration Impact Factor'''
        
        # behavioral collaborative impacts between personalized patterns by wasserstein_distance
        Weight_user_behavior = -wasserstein_distance_matmul(P_user_behavior_m, P_user_behavior_c, P_user_behavior_m, P_user_behavior_c)

        # corresponding behavior-relation representation
        bb_m = torch.zeros(u_id.size(0),self.n_b+1,self.n_b+1,self.d_model).cuda()
        bb_c = torch.zeros(u_id.size(0),self.n_b+1,self.n_b+1,self.d_model).cuda()

        for i in range(self.n_b):
            for j in range(self.n_b):
                bb_m[:,i+1,j+1,:] = torch.matmul(Weight_user_behavior[:,i+1,j+1].unsqueeze(1),self.bb_embedding_m(torch.LongTensor([i*4+j+1]).cuda()))
                bb_c[:,i+1,j+1,:] = torch.matmul(Weight_user_behavior[:,i+1,j+1].unsqueeze(1),self.bb_embedding_c(torch.LongTensor([i*4+j+1]).cuda()))
        
        bb_c = self.activation(bb_c) + 1

        # running over multiple transformer blocks using ''' Fused Behavior-Aware Attention '''
        for transformer in self.transformer_blocks:
            x_m, x_c, W_probs = transformer.forward(x_m, x_c, b_seq , b_m, b_c, bb_m, bb_c, p_m, p_c, mask)
        
        # Pattern-aware next-item prediction
        x_m,x_c = SAGP( x_m, self.WPub(P_user_behavior_m[torch.arange(batch_size)[:,None],b_seq[torch.arange(batch_size)],:]),
                        x_c, P_user_behavior_c[torch.arange(batch_size)[:,None],b_seq[torch.arange(batch_size)],:])

        return x_m, x_c, W_probs
