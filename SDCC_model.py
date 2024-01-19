import torch
import torch.nn as nn
import torch.nn.functional as F
from objectives import dcp_loss, mdcp_loss, l21_loss

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
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        o_h = x
        o = self.emb2(x)

        return o, o_h

class Decoder(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Decoder, self).__init__()
        self.emb1 = nn.Linear(in_size, hidden_size)
        self.blk1 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk2 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.blk3 = MlpBlock(hidden_size, hidden_size, hidden_size)
        self.emb2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.emb1(x)
        o_h = x
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        o = self.emb2(x)

        return o, o_h

class DDC(nn.Module):
    def __init__(self, input_dim, n_clusters, n_features=512):
        """
        DDC clustering module

        :param input_dim: Shape of inputs.
        :param cfg: DDC config. See `config.defaults.DDC`
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, n_features, bias=True),
            nn.GELU(),
            nn.Linear(n_features, n_features, bias=True),
            nn.GELU(),
            nn.Linear(n_features, n_features, bias=True),
            nn.GELU(),
            nn.BatchNorm1d(num_features=n_features)
        )
        self.output = nn.Sequential(nn.Linear(n_features, n_clusters), nn.Softmax(dim=1))

    def forward(self, x):
        hidden = self.encoder(x)
        output = self.output(hidden)
        return output, hidden

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

class DCP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, view, cca_dim, r, device, n_iter=15, dcc_solver='pytorch', dcc_dtype=torch.float32, dec_loss_type='l21', twoview=False):
        super(DCP, self).__init__()
        self.Enc = Encoder(in_size, hidden_size, out_size)
        self.Dec = Decoder(out_size, hidden_size, in_size)

        self.fusion = nn.Sequential(
            nn.Linear(cca_dim*view, cca_dim, bias=True),
        )
        self.view = view
        self.cca_dim = cca_dim
        self.device = device
        self.dec_loss_type = dec_loss_type
        self.twoview = twoview

        if twoview:
            self.dcp_loss = dcp_loss(cca_dim, r, device, dcc_dtype).loss
        else:
            if dcc_solver == 'numpy':
                self.dcp_loss = mdcp_loss(cca_dim, r, device, n_iter, dcc_dtype).loss_numpy_fast
            else:
                self.dcp_loss = mdcp_loss(cca_dim, r, device, n_iter, dcc_dtype).loss
        if dec_loss_type == 'l21':
            self.l21_loss = l21_loss().loss
        elif dec_loss_type == 'mse':
            self.mse_loss = nn.MSELoss(reduction='mean')
        else:
            raise "unknown decoder loss"

    def forward(self, x):
        eo = []
        eh = []
        do = []
        dh = []

        dec_loss = 0
        for v in range(self.view):
            eot, eoh = self.Enc(x[v])
            eo.append(eot)
            eh.append(eoh)

            dot, doh = self.Dec(eot)
            do.append(dot)
            dh.append(doh)

            dec_loss += self.dec_loss(x[v], eoh, dot, doh)

        dcp_loss, W = self.dcp_loss(eo)
        c_list = []
        for v in range(self.view):
            wt = torch.tensor(W[v], dtype=torch.float32, device=self.device)
            ct = torch.matmul(wt.t(), x[v])
            c_list.append(ct)
        tfused = torch.cat(c_list, dim=0)
        fused = self.fusion(tfused.t())
        return eo, fused, dcp_loss, dec_loss

    def dec_loss(self, x, eh, do, dh):
        if self.dec_loss_type == 'l21':
            loss1 = self.l21_loss(x, do)
            loss2 = self.l21_loss(eh, dh)
        elif self.dec_loss_type == 'mse':
            loss1 = self.mse_loss(x, do)
            loss2 = self.mse_loss(eh, dh)
        else:
            raise "unknown decoder loss"

        loss = loss1 + loss2
        return loss
