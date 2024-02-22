
import torch
import pytorch_lightning as pl
from .models import WassersteinPredictionHead
from .models.bert4rec import BERT
from .utils import recalls_and_ndcgs_for_ks

class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: BERT,
        ):
        super().__init__()
        self.backbone = backbone
        self.n_b = backbone.n_b
        self.max_len = backbone.max_len

        self.head = WassersteinPredictionHead(backbone.d_model, backbone.num_items, self.backbone.item_embedding_m.token,self.backbone.item_embedding_c.token)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, input_ids, b_seq, u_id):
        return self.backbone(input_ids, b_seq, u_id)
        

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        user_id = batch['user_id']
        #outputs = self(input_ids, b_seq, user_id)
        outputsm, outputsc ,W_pro = self(input_ids, b_seq ,user_id)
        outputsm = outputsm.view(-1, outputsm.size(-1))  # BT x H
        outputsc = outputsc.view(-1, outputsc.size(-1))

        labels = batch['labels']
        labels = labels.view(-1)  # BT

        valid = labels>0
        valid_index = valid.nonzero().squeeze()  # M
        valid_outputsm = outputsm[valid_index]
        valid_outputsc = outputsc[valid_index]

        valid_b_seq = b_seq.view(-1)[valid_index] # M
        valid_labels = labels[valid_index]
        #valid_logits = self.head(valid_outputs, valid_b_seq) # M
        valid_logits = self.head(valid_outputsm, valid_outputsc, valid_b_seq)
        loss = self.loss(valid_logits, valid_labels)
        loss = loss.unsqueeze(0)
        return {'loss':loss}

        
    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        user_id = batch['user_id']
        outputsm, outputsc ,W_pro = self(input_ids, b_seq ,user_id)

        # get scores (B x C) for evaluation
        last_outputsm = outputsm[:, -1, :]
        last_outputsc = outputsc[:, -1, :]
        last_b_seq = b_seq[:,-1]
        candidates = batch['candidates'].squeeze() # B x C
        logits = self.head(last_outputsm, last_outputsc, last_b_seq, candidates)
        labels = batch['labels'].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

        return metrics
    
    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())