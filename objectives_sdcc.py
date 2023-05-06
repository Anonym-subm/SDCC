import torch
import torch.nn as nn
import numpy as np
import scipy
from torch import linalg as LA
from kernel import *
EPSILON = 1E-9

class dcc_loss():
    def __init__(self, dim, r, device):
        self.dim = dim
        self.r = r
        self.device = device

    def loss(self, H):
        H1 = H[0]
        H2 = H[1]

        r1 = self.r
        r2 = self.r
        eps = 1e-9

        o1 = H1.size(0)
        o2 = H2.size(0)

        m = H1.size(1)

        H1bar = H1
        H2bar = H2

        SigmaHat12 = torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device, dtype=H1.dtype)
        SigmaHat22 = torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device, dtype=H2.dtype)

        [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
        [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)

        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        U, S, V = torch.linalg.svd(Tval)
        S = torch.where(S > eps, S, (torch.ones(S.shape) * eps).to(self.device))
        S = S.topk(self.dim)[0]
        corr = torch.sum(S)

        W = []
        W1 = torch.matmul(SigmaHat11RootInv, U[:, 0:self.dim])
        W2 = torch.matmul(SigmaHat22RootInv, V[:, 0:self.dim])
        W.append(W1.detach().cpu().numpy())
        W.append(W2.detach().cpu().numpy())
        return -corr, W

class mdcc_loss():
    def __init__(self, dim, r, device=torch.device('cuda')):
        self.dim = dim
        self.r = r
        self.device = device

    def loss(self, X):
        v = len(X)
        N = X[0].shape[1]

        npX = []
        d_list = []
        for i in range(v):
            npX.append(X[i].cpu().detach().numpy())
            d_list.append(npX[i].shape[0])
        d_sum = sum(d_list)
        A = np.zeros((d_sum, d_sum))

        siiRootInv = []
        for i in range(v):
            sii = np.dot(npX[i], npX[i].T)
            sii = sii + self.r * np.identity(d_list[i])
            siiInv = np.linalg.inv(sii)
            srinv = scipy.linalg.sqrtm(siiInv)
            srinv = srinv.real
            siiRootInv.append(srinv)

        for i in range(v):
            di = d_list[i]
            si = sum(d_list[0:i])
            for j in range(v):
                dj = d_list[j]
                sj = sum(d_list[0:j])
                sij = np.dot(npX[i], npX[j].T)
                A[si:si+di, sj:sj+dj] = np.dot(np.dot(
                    siiRootInv[i], sij), siiRootInv[j])

        V = []

        tol = 1e-3
        n_iter = 15

        # Initialize V
        for i in range(v):
            V.append(np.ones((d_list[i], self.dim)))
            for k in range(self.dim):
                V[i][:, k] = V[i][:, k]/np.linalg.norm(V[i][:, k])

        for k in range(self.dim):
            if k == 0:
                S = A
            else:
                W = np.zeros((d_sum, v*k))
                for i in range(v):
                    di = d_list[i]
                    si = sum(d_list[0:i])
                    for j in range(k):
                        W[si:si+di, i*k+j] = V[i][:, j]
                S = (A - np.dot(np.dot(W, W.T), A))

            for n in range(n_iter):
                for i in range(v):
                    di = d_list[i]
                    si = sum(d_list[0:i])
                    y = np.zeros(di)
                    for j in range(v):
                        dj = d_list[j]
                        sj = sum(d_list[0:j])
                        t = np.dot(S[si:si+di, sj:sj+dj], V[j][:, k])
                        y = y + t
                    lam = np.linalg.norm(y)
                    if lam != 0:
                        V[i][:, k] = y / lam

        W = []
        Wn = []
        for i in range(v):
            t = np.dot(siiRootInv[i], V[i])
            Wn.append(t)
            W.append(torch.tensor(t, device=self.device, dtype=torch.float32))

        corr = 0
        for i in range(v):
            for j in range(v):
                sij = torch.matmul(X[i], X[j].t())
                corr += torch.matmul(
                    torch.matmul(W[i][:, 1].t(), sij), W[j][:, 1]
                )

        return -torch.abs(corr), Wn

class safe_loss(nn.Module):
    def __init__(self, class_num, device):
        super(safe_loss, self).__init__()
        self.class_num = class_num
        self.device = device

    def forward_cluster(self, hidden, output, print_sign=False):
        hidden_kernel = vector_kernel(hidden, rel_sigma=0.15)
        l1 = self.DDC1(output, hidden_kernel, self.class_num)
        l2 = self.DDC2(output)
        l3 = self.DDC3(self.class_num, output, hidden_kernel)
        if print_sign:
            print(l1.item())
            print(l2.item())
            print(l3.item())
        return l1+l2+l3, l1.item() + l2.item() + l3.item()

    "Adopted from https://github.com/DanielTrosten/mvc"

    def triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _atleast_epsilon(self, X, eps=EPSILON):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: th.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: th.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  th.Tensor
        :param K: Kernel matrix
        :type K: th.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: th.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=EPSILON ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self.triu(nom / torch.sqrt(dnom_squared))
        return d

    # ======================================================================================================================
    # Loss terms
    # ======================================================================================================================

    def DDC1(self, output, hidden_kernel, n_clusters):
        """
        L_1 loss from DDC
        """
        # required_tensors = ["hidden_kernel"]
        return self.d_cs(output, hidden_kernel, n_clusters)

    def DDC2(self, output):
        """
        L_2 loss from DDC
        """
        n = output.size(0)
        return 2 / (n * (n - 1)) * self.triu(output @ torch.t(output))

    def DDC2Flipped(self, output, n_clusters):
        """
        Flipped version of the L_2 loss from DDC. Used by EAMC
        """

        return 2 / (n_clusters * (n_clusters - 1)) * self.triu(torch.t(output) @ output)

    def DDC3(self, n_clusters, output, hidden_kernel):
        """
        L_3 loss from DDC
        """

        eye = torch.eye(n_clusters, device=self.device)

        m = torch.exp(-cdist(output, eye))
        return self.d_cs(m, hidden_kernel, n_clusters)