from objectives import safe_loss
from SDCC_model import *
from utils import *
import time
import logging, math
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle
import numpy as np
seed_list = np.random.default_rng(seed=0).choice(10000, size=10, replace=False)

logging.getLogger('matplotlib.font_manager').disabled = True

dcc_solver = 'pytorch'

config = dict(
    n_view=5,
    lr=0.0001,
    reg_par=0.001,
    n_fea=512,

    dim=40,
    out_dim=2,
    lmbda=0.1,
    lmbda2=0.01,
    lmbda3=0.01,
    r=1e-1,
    n_iter=10,
    t_time=0,
    dcc_dtype=64,
    dec_loss='l21',

    dataset = 'Caltech-5v',
)

class Solver():
    def __init__(self, old_model, new_model, gate, ddc_model, lmbda, lmbda2, lmbda3, dim, n_class, epoch_num, batch_size,
                 learning_rate, reg_par, r, device=torch.device('cpu')):
        self.old_model = old_model.to(device)
        self.new_model = new_model.to(device)
        self.gate = gate.to(device)
        self.ddc_model = ddc_model.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.lmbda3 = lmbda3
        self.n_class = n_class
        self.cluster_loss = safe_loss(n_class, device)
        self.sdcp_optimizer = torch.optim.Adam(list(self.old_model.parameters()) + list(self.new_model.parameters())
                                               + list(self.gate.parameters()) + list(self.ddc_model.parameters()),
                                               lr=learning_rate, weight_decay=reg_par)
        self.finetune_optimizer = torch.optim.Adam(list(self.gate.parameters()) + list(self.ddc_model.parameters()),
                                               lr=learning_rate, weight_decay=reg_par)
        self.finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.finetune_optimizer, T_max=150,
                                                eta_min=learning_rate/10)
        self.device = device

        self.dim = dim
        self.r = r

        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")

    def fit(self, x, lbl, times=0):
        n_view = len(x)
        xt = []
        self.npx = []
        for v in range(n_view):
            x[v] = x[v].to(self.device)
            xt.append(x[v].t())
            self.npx.append(x[v].cpu().numpy())

        self.nmi = 0
        self.acc = 0
        self.pur = 0
        self.ep = self.epoch_num

        self.full_batch = True

        loss_list = []
        nmi_list = []
        acc_list = []
        pur_list = []
        for epoch in range(self.epoch_num):
            train_losses = []
            epoch_start_time = time.time()

            self.old_model.train()
            self.new_model.train()
            self.gate.train()
            self.ddc_model.train()
            self.sdcp_optimizer.zero_grad()

            batch_x = xt

            eo_old, fused_old, dcp_loss_old, dec_loss_old = self.old_model(batch_x)
            eo_new, fused_new, dcp_loss_new, dec_loss_new = self.new_model(batch_x)
            fused = self.gate([fused_old, fused_new])
            dcp_loss = dcp_loss_old + dcp_loss_new
            dec_loss = dec_loss_old + dec_loss_new

            output, hidden = self.ddc_model(fused)

            pred_vector = output.detach().cpu().numpy()
            clbl = np.argmax(np.array(pred_vector), axis=1)

            safe_loss, _ = self.cluster_loss.forward_cluster(hidden, output)
            loss = self.lmbda * dcp_loss + self.lmbda2 * dec_loss + self.lmbda3 * safe_loss

            if math.isnan(loss.item()):
                return self.nmi, self.acc, self.pur, self.ep

            train_losses.append(loss.item())
            loss.backward()
            self.sdcp_optimizer.step()

            train_loss = np.mean(train_losses)

            info_string = "{:d} Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f}"
            epoch_time = time.time() - epoch_start_time
            loss_list.append(train_loss)
            self.logger.info(info_string.format(
                times, epoch + 1, self.epoch_num, epoch_time, train_loss))

        self.old_model.eval()
        self.new_model.eval()
        batch_x = xt
        with torch.no_grad():
            eo_old_f, fused_old_f, dcp_loss_old_f, dec_loss_old_f = self.old_model(batch_x)
            eo_new_f, fused_new_f, dcp_loss_new_f, dec_loss_new_f = self.new_model(batch_x)
        for epoch in range(150):
            train_losses = []
            epoch_start_time = time.time()

            self.gate.train()
            self.ddc_model.train()
            self.finetune_optimizer.zero_grad()

            fused_old = fused_old_f.clone().detach()
            fused_new = fused_new_f.clone().detach()
            fused = self.gate([fused_old, fused_new])

            output, hidden = self.ddc_model(fused)

            pred_vector = output.detach().cpu().numpy()
            clbl = np.argmax(np.array(pred_vector), axis=1)

            safe_loss, _ = self.cluster_loss.forward_cluster(hidden, output)
            loss = safe_loss

            if math.isnan(loss.item()):
                return self.nmi, self.acc, self.pur, self.ep

            train_losses.append(loss.item())
            loss.backward()
            self.finetune_optimizer.step()
            self.finetune_scheduler.step()

            nmi_score, pur_score, acc_score = cluster_eval(y_true=lbl, y_pred=clbl)
            nmi_list.append(nmi_score)
            pur_list.append(pur_score)
            acc_list.append(acc_score)

            if self.nmi < nmi_score:
                self.nmi = nmi_score
                self.acc = acc_score
                self.pur = pur_score
                self.ep = epoch + 1

            train_loss = np.mean(train_losses)

            info_string = "{:d} Epoch {:d}/{:d} - lr: {:.5f} - time: {:.2f} - training_loss: {:.4f}"
            epoch_time = time.time() - epoch_start_time
            loss_list.append(train_loss)
            self.logger.info(info_string.format(
                times, epoch + 1, 150, self.finetune_optimizer.param_groups[0]['lr'], epoch_time, train_loss))

        return self.nmi, self.acc, self.pur, self.ep


def main(X, lbl, N_sample, N_sam_fea, n_class, times=0):
    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    n_view = config['n_view']

    outdim_size = config['out_dim']
    dim = config['dim']

    learning_rate = config['lr']
    batch_size = max(N_sam_fea)

    r = config['r']
    n_iter = config['n_iter']

    cca_dim = min(dim, min(N_sam_fea))

    reg_par = config['reg_par']

    input_size = N_sample
    hidden_size = config['n_fea']
    output_size = outdim_size

    if config['dcc_dtype'] == 32:
        dcc_dtype = torch.float32
    elif config['dcc_dtype'] == 64:
        dcc_dtype = torch.float64
    else:
        dcc_dtype = torch.float32
    dec_loss_type = config['dec_loss']
    epoch_num = 21

    old_model = DCP(input_size, hidden_size, output_size, n_view - 1, cca_dim, r, device, n_iter=n_iter, dcc_solver=dcc_solver, dcc_dtype=dcc_dtype, dec_loss_type=dec_loss_type)
    new_model = DCP(input_size, hidden_size, output_size, n_view, cca_dim, r, device, n_iter=n_iter, dcc_solver=dcc_solver, dcc_dtype=dcc_dtype, dec_loss_type=dec_loss_type)

    gate = WeightedMean(2)
    ddc_model = DDC(cca_dim, n_class)

    lmbda = config['lmbda']
    lmbda2 = config['lmbda2']
    lmbda3 = config['lmbda3']

    solver = Solver(old_model, new_model, gate, ddc_model, lmbda, lmbda2, lmbda3, dim, n_class, epoch_num, batch_size,
                    learning_rate, reg_par, r, device)

    nmi, acc, pur, ep = solver.fit(X, lbl, times=times)

    return nmi, pur, acc, ep

if __name__ == '__main__':
    t_time = config['t_time']
    dataset = config['dataset']
    n_view = config['n_view']

    X, lbl, N_sample, N_sam_fea, n_class = load_mv_dataset(dataset, n_view)

    t_idx = t_time
    SEED = seed_list[t_idx]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    nmi, pur, acc, ep = main(X, lbl, N_sample, N_sam_fea, n_class, times=t_idx)

    print("=====================================================")
    print('Final: NMI score: {:.2f}%, PUR score: {:.2f}%, ACC score: {:.2f}%'
          .format(nmi * 100, pur * 100, acc * 100))
