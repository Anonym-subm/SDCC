import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from objectives_sdcc import dcc_loss, mdcc_loss

class MlpBlock(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super(MlpBlock, self).__init__()
        self.linear1 = nn.Linear(d_in, hidden)
        self.linear2 = nn.Linear(hidden, d_out)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(d_in)

    def forward(self, src):
        x = src
        x = x + self.linear2(self.gelu(self.linear1(self.layer_norm(x))))
        return x


class Encoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Encoder, self).__init__()
        self.emb1 = nn.Linear(in_size, hidden_size)
        self.blk1 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk2 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk3 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.emb2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.emb1(x)
        o1 = x
        x = self.blk1(x)
        o2 = x
        x = self.blk2(x)
        o3 = x
        x = self.blk3(x)
        o = self.emb2(x)

        return o, o1, o2, o3

class Decoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, drop_prob=0.1):
        super(Decoder, self).__init__()
        self.emb1 = nn.Linear(in_size, hidden_size)
        self.blk1 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk2 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk3 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.emb2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.emb1(x)
        x = self.blk1(x)
        o1 = x
        x = self.blk2(x)
        o2 = x
        x = self.blk3(x)
        o3 = x
        o = self.emb2(x)

        return o, o1, o2, o3

class WeightedMean(nn.Module):
    """
    Weighted mean fusion.

    :param cfg: Fusion config. See config.defaults.Fusion
    :param input_sizes: Input shapes
    """
    def __init__(self, n_views):
        super().__init__()
        self.n_views = n_views
        self.weights = nn.Parameter(torch.full((self.n_views,), 1 / self.n_views), requires_grad=True)

    def forward(self, inputs):
        return _weighted_sum(inputs, self.weights, normalize_weights=True)

def _weighted_sum(tensors, weights, normalize_weights=True):
    if normalize_weights:
        weights = F.softmax(weights, dim=0)
    out = torch.sum(weights[None, None, :] * torch.stack(tensors, dim=-1), dim=-1)
    return out

class DCC(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, view, cca_dim, r, device):
        super(DCC, self).__init__()
        self.Enc = Encoder(in_size, hidden_size, out_size)
        self.Dec = Decoder(out_size, hidden_size, in_size)
        self.view = view
        self.cca_dim = cca_dim
        self.device = device

        self.dcc_loss = mdcc_loss(cca_dim, r, device).loss
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, x):
        eo = []

        dec_loss = 0
        for v in range(self.view):
            eot, eot1, eot2, eot3 = self.Enc(x[v])
            eo.append(eot)

            dot, dot1, dot2, dot3 = self.Dec(eot)

            dec_loss += self.dec_loss(x[v], eot1, eot2, eot3, dot, dot1, dot2, dot3)

        dcc_loss, W = self.dcc_loss(eo)
        N_sample = x[0].size(1)
        c_list = []
        fused = None
        for v in range(self.view):
            wt = torch.tensor(W[v], dtype=torch.float32, device=self.device)
            ct = torch.matmul(wt.t(), x[v])
            c_list.append(ct)
            if fused is None:
                fused = ct
            else:
                fused = fused + ct
        return eo, fused, dcc_loss, dec_loss

    def dec_loss(self, x, eo1, eo2, eo3, do, do1, do2, do3):
        sample_size = x.size(1)

        loss1 = self.mse_loss(x, do)
        loss2 = self.mse_loss(eo1, do3)
        loss3 = self.mse_loss(eo2, do2)
        loss4 = self.mse_loss(eo3, do1)

        loss = loss1 + loss2 + loss3 + loss4
        return loss

