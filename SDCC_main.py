from objectives_sdcc import safe_loss
from kmeans_pytorch import kmeans
from SDCC_model import *
from utils import *
import time, os, csv
import logging
try:
    import cPickle as thepickle
except ImportError:
    import _pickle as thepickle

import numpy as np
import wandb
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
logging.getLogger('matplotlib.font_manager').disabled = True

isSave = False
isLoad = False
isDebug = False

defconfig = dict(         # Caltech 5v
    n_view=5,
    lr=0.001,
    reg_par=0.001,
    n_fea=512,
    drop_prob=0.1,

    dim=24,
    out_dim=44,
    ep_num=80,
    lmbda=1e-6,
    lmbda2=1e-5,
    r=1e-2,

    dataset = 'Caltech-5V',
)

wandb.init(config=defconfig, mode='disabled')
config = wandb.config

class Solver():
    def __init__(self, old_model, new_model, gate, lmbda, lmbda2, dim, n_class, epoch_num, batch_size,
                 learning_rate, reg_par, r, device=torch.device('cpu')):
        self.old_model = old_model.to(device)
        self.new_model = new_model.to(device)
        self.gate = gate.to(device)
        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lmbda = lmbda
        self.lmbda2 = lmbda2
        self.safe_loss = safe_loss(n_class, device)
        self.sdcc_optimizer = torch.optim.Adam(list(self.old_model.parameters()) + list(self.new_model.parameters())
                                               + list(self.gate.parameters()),
                                               lr=learning_rate, weight_decay=reg_par)
        self.device = device

        self.dim = dim
        self.r = r

        formatter = logging.Formatter(
            "[ %(levelname)s : %(asctime)s ] - %(message)s")
        logging.basicConfig(
            level=logging.DEBUG, format="[ %(levelname)s : %(asctime)s ] - %(message)s")
        self.logger = logging.getLogger("Pytorch")

        self.logger.info(self.new_model)
        self.logger.info(self.gate)
        self.logger.info("dim: " + str(config.dim))

    def fit(self, x, lbl):
        n_view = len(x)
        xt = []
        self.npx = []
        for v in range(n_view):
            x[v] = x[v].to(self.device)
            xt.append(x[v].t())
            self.npx.append(x[v].cpu().numpy())

        self.best_nmi = 0
        self.best_acc = 0
        self.best_pur = 0
        self.best_ep = self.epoch_num

        self.full_batch = True

        if isLoad:
            self.sdcc_model = torch.load('./model/sdcc_model_' + config.dataset_name + '.pth')
        else:
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
                self.sdcc_optimizer.zero_grad()
                batch_x = xt

                eo_old, fused_old, dcc_loss_old, dec_loss_old = self.old_model(batch_x)
                eo_new, fused_new, dcc_loss_new, dec_loss_new = self.new_model(batch_x)
                fused = self.gate([fused_old, fused_new])
                dcc_loss = self.gate([dcc_loss_old, dcc_loss_new])
                dec_loss = self.gate([dec_loss_old, dec_loss_new])

                pred, cluster_centers = kmeans(
                    X=fused.t(), num_clusters=n_class, distance='euclidean', device=self.device, seed=0, tqdm_flag=False
                )
                clbl = pred.detach().cpu().numpy()

                pred = F.one_hot(pred).float()

                safe_loss, _ = self.safe_loss.forward_cluster(fused.t(), pred)
                loss = self.lmbda * dcc_loss + self.lmbda2 * dec_loss + safe_loss

                train_losses.append(loss.item())
                loss.backward()
                self.sdcc_optimizer.step()


                nmi_score, pur_score, acc_score = cluster_eval(y_true=lbl, y_pred=clbl)
                nmi_list.append(nmi_score)
                pur_list.append(pur_score)
                acc_list.append(acc_score)

                if self.best_nmi < nmi_score:
                    self.best_nmi = nmi_score
                    self.best_acc = acc_score
                    self.best_pur = pur_score
                    self.best_ep = epoch + 1

                train_loss = np.mean(train_losses)

                info_string = "Epoch {:d}/{:d} - time: {:.2f} - training_loss: {:.4f} nmi: {:.4f} acc: {:.4f} pur: {:.4f}"
                epoch_time = time.time() - epoch_start_time
                loss_list.append(train_loss)
                self.logger.info(info_string.format(
                    epoch + 1, self.epoch_num, epoch_time, train_loss, nmi_score, acc_score, pur_score))
                wandb.log({
                    "Train Loss": train_loss,
                    "ep_nmi": nmi_score,
                    "ep_acc": acc_score,
                    "ep_pur": pur_score,
                }, step=epoch + 1)


        return self.best_nmi, self.best_acc, self.best_pur, self.best_ep


if __name__ == '__main__':
    ############
    # Parameters Section
    dataset = config.dataset
    device = torch.device('cuda')
    print("Using", torch.cuda.device_count(), "GPUs")

    n_view = config.n_view

    dpca1 = dpca2 = None
    try:
        dpca1 = config.dpca1
        dpca2 = config.dpca2
    except:
        pass

    X, lbl, N_sample, N_sam_fea, n_class = load_mv_dataset(dataset, n_view)

    outdim_size = config.out_dim
    dim = config.dim

    learning_rate = config.lr
    epoch_num = config.ep_num
    batch_size = max(N_sam_fea)

    r = 1e-8
    try:
        r = config.r
    except:
        pass

    cca_dim = min(dim, min(N_sam_fea))
    n_fea = config.n_fea

    reg_par = config.reg_par

    input_size = N_sample
    hidden_size = config.n_fea
    output_size = outdim_size

    old_model = DCC(input_size, hidden_size, output_size, n_view - 1, cca_dim, r, device)
    new_model = DCC(input_size, hidden_size, output_size, n_view, cca_dim, r, device)

    gate = WeightedMean(2)

    lmbda = config.lmbda
    lmbda2 = config.lmbda2

    solver = Solver(old_model, new_model, gate, lmbda, lmbda2, dim, n_class, epoch_num, batch_size,
                    learning_rate, reg_par, r, device)

    if isDebug:
        best_nmi, best_acc, best_pur, best_ep = solver.fit(X, lbl)
    else:
        try:
            best_nmi, best_acc, best_pur, best_ep = solver.fit(X, lbl)
        except:
            best_nmi = solver.best_nmi
            best_acc = solver.best_acc
            best_pur = solver.best_pur
            best_ep = solver.best_ep

    if isSave:
        torch.save(solver.sdcc_model, './model/sdcc_model_' + config.dataset_name + '.pth')

    print('Combined: best epoch: ' + str(best_ep) + ' best NMI score: ' + str(best_nmi) +
          ' Purity score: ' + str(best_pur) + ' Accuracy score: ' + str(best_acc))

    wandb.log({
        "NMI": best_nmi,
        "Purity": best_pur,
        "Accuracy": best_acc,
        "Best_epoch": best_ep
    })

