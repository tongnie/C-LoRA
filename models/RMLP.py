import torch
import torch.nn as nn
from torch.nn import functional as F

import torch
import torch.nn as nn
from layers.Embed import DataEmbedding_inverted
from einops import rearrange, einsum, repeat

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.d_model)
        )

        self.projection = nn.Linear(configs.d_model, configs.pred_len)

        # self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.enc_in)
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name == 'classification' or self.task_name == 'anomaly_detection' or self.task_name == 'imputation':
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        # Embedding
        embed_dim = configs.d_model - configs.node_dim
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, embed_dim, configs.embed, configs.freq,
                                                    configs.dropout)
        self.rank = configs.rank
        self.node_dim = configs.node_dim
        ## Channel-wise LoRA
        self.adapter = nn.Parameter(torch.empty(configs.enc_in, embed_dim, self.rank))  # [n,D,d]
        nn.init.xavier_uniform_(self.adapter)
        self.lora = nn.Linear(self.rank, configs.node_dim, bias=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None

    def forecast(self, x_enc):
        # x: [B, L, D]
        x_enc = self.rev(x_enc, 'norm')

        B, _, N = x_enc.shape
        # Embedding
        x_enc = self.enc_embedding(x_enc, None)  # [b n d]
        ## Local adaptation
        adaptation = []
        adapter = F.relu(self.lora(self.adapter))  # [n D d]
        adapter = adapter.permute(1, 2, 0)  # [n D d] -> [D d n]
        adapter = repeat(adapter, 'D d n -> repeat D d n', repeat=B)  # [b, D, d, n]
        x_enc = x_enc.transpose(1, 2)  # [b n D] -> [b D n]
        adaptation.append(torch.einsum('bDn,bDdn->bdn', [x_enc, adapter]))  # [B, d, N]
        x_enc = torch.cat([x_enc] + adaptation, dim=1)  # [B, H', N]
        ## Local adaptation

        x_enc = x_enc + self.temporal(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm')

        return pred
        
