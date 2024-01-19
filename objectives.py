import torch
import torch.nn as nn
import numpy as np
import scipy
from torch import linalg as LA
from kernel import *
import time
EPSILON = 1E-9

class dcp_loss():
    def __init__(self, dim, r, device, dtype=torch.float32):
        self.dim = dim
        self.r = r
        self.device = device
        self.dtype = dtype

    def loss(self, H):
        H1 = H[0]
        H2 = H[1]

        r1 = self.r
        r2 = self.r
        eps = 1e-9

        o1 = H1.size(0)
        o2 = H2.size(0)

        with torch.no_grad():
            if self.dtype == torch.float64:
                H1bar = H1.double()
                H2bar = H2.double()
            else:
                H1bar = H1
                H2bar = H2

            SigmaHat12 = torch.matmul(H1bar, H2bar.t())
            SigmaHat11 = torch.matmul(H1bar, H1bar.t()) + r1 * torch.eye(o1, device=self.device, dtype=self.dtype)
            SigmaHat22 = torch.matmul(H2bar, H2bar.t()) + r2 * torch.eye(o2, device=self.device, dtype=self.dtype)

            [D1, V1] = torch.linalg.eigh(SigmaHat11)
            [D2, V2] = torch.linalg.eigh(SigmaHat22)

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
        if self.dtype == torch.float64:
            corr = corr.float()

        W = []
        W1 = torch.matmul(SigmaHat11RootInv, U[:, 0:self.dim])
        W2 = torch.matmul(SigmaHat22RootInv, V[:, 0:self.dim])
        W.append(W1.detach().cpu().numpy())
        W.append(W2.detach().cpu().numpy())

        return -corr, W

class mdcp_loss():
    def __init__(self, dim, r, device=torch.device('cuda'), n_iter=15, dcc_dtype=torch.float32):
        self.dim = dim
        self.r = r
        self.device = device
        self.n_iter = n_iter
        self.eps = 1e-6
        self.dtype = dcc_dtype

    def loss(self, X):
        dtype = self.dtype

        v = len(X)

        with torch.no_grad():
            d_list = []
            for i in range(v):
                d_list.append(X[i].shape[0])
            d_sum = sum(d_list)
            A = torch.zeros(d_sum, d_sum, device=self.device, dtype=dtype)

            ts = time.time()
            siiRootInv = []
            for i in range(v):
                sii = torch.matmul(X[i], X[i].t())
                sii = sii + self.r * torch.eye(d_list[i], device=self.device, dtype=dtype)
                [D, V] = torch.linalg.eigh(sii)
                idx = D > self.eps
                V = V[:, idx]
                D = D[idx]
                srinv = torch.matmul(torch.matmul(
                    V, torch.diag(D ** -0.5)), V.t()
                )
                siiRootInv.append(srinv)

            ts2 = time.time()
            for i in range(v):
                di = d_list[i]
                si = sum(d_list[0:i])
                for j in range(v):
                    dj = d_list[j]
                    sj = sum(d_list[0:j])

                    if dtype == torch.float64:
                        sij = torch.matmul(X[i], X[j].t()).double()
                    else:
                        sij = torch.matmul(X[i], X[j].t())
                    A[si:si+di, sj:sj+dj] = torch.matmul(torch.matmul(
                        siiRootInv[i], sij), siiRootInv[j]
                    )

            V = torch.ones(d_sum, self.dim, device=self.device, dtype=dtype)

            n_iter = self.n_iter

            ts3 = time.time()
            for i in range(v):
                di = d_list[i]
                si = sum(d_list[0:i])
                for k in range(self.dim):
                    V[si:si+di, k] = V[si:si+di, k] / torch.linalg.vector_norm(V[si:si+di, k])

            t1 = 0
            t2 = 0
            for k in range(self.dim):
                t11 = time.time()
                if k == 0:
                    S = A
                else:
                    W = torch.zeros(d_sum, v*k, device=self.device, dtype=dtype)
                    for i in range(v):
                        di = d_list[i]
                        si = sum(d_list[0:i])
                        W[si:si+di, i*k:(i+1)*k] = V[si:si+di, 0:k]
                    S = (A - torch.matmul(torch.matmul(W, W.t()), A))
                t1 += time.time() - t11

                t22 = time.time()
                for n in range(n_iter):
                    for i in range(v):
                        di = d_list[i]
                        si = sum(d_list[0:i])

                        y = torch.matmul(S[si:si+di, :], V[:, k])
                        ilam = torch.pow(torch.sum(torch.pow(y, 2)), -0.5)
                        V[si:si + di, k] = y * ilam

                t2 += time.time() - t22

            W = []
            Wn = []
            for i in range(v):
                di = d_list[i]
                si = sum(d_list[0:i])
                t = torch.matmul(siiRootInv[i], V[si:si+di, :])
                Wn.append(t.cpu().detach().numpy())
                W.append(t.float())

        corr = 0
        for i in range(v):
            for j in range(v):
                sij = torch.matmul(X[i], X[j].t())
                corr += torch.matmul(
                    torch.matmul(W[i][:, 1].t(), sij), W[j][:, 1]
                )

        return -torch.abs(corr), Wn

class l21_loss():
    def __init__(self):
        pass

    def loss(self, H1, H2):
        A = H1 - H2

        norm2 = LA.norm(A, 2, dim=1)
        norm = LA.norm(norm2, 1, dim=0)

        return norm

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

    def DDC1(self, output, hidden_kernel, n_clusters):
        """
        L_1 loss from DDC
        """
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