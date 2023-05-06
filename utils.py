import gzip
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn import metrics, neighbors
from metrics import accuracy, purity
import torch

def load_data_from_mat(data_file, dpca1=None, dpca2=None, dtype=torch.float32):
    print('loading data ...')
    data_dict = loadmat(data_file)
    X1 = torch.tensor(data_dict['X1'], dtype=dtype)
    X2 = torch.tensor(data_dict['X2'], dtype=dtype)

    if dpca1 is not None:
        X1 = X1[:, 0:min(dpca1, X1.size(1))]
        X2 = X2[:, 0:min(dpca2, X2.size(1))]

    lbl = data_dict['Label'].T
    lbl = lbl[0]
    if min(np.unique(lbl)) == 1:
        lbl = lbl - 1
    return X1, X2, lbl

def load_mv_dataset(dataset, n_view, dtype=torch.float32):
    print('loading data ...')
    if dataset.startswith('Caltech'):
        path = './dataset/Caltech/Caltech_pca98.mat'
    elif dataset.startswith('CCV'):
        path = './dataset/CCV/CCV_pca98.mat'
    elif dataset.startswith('NGs'):
        path = './dataset/NGs/NGs_pca98.mat'
    elif dataset.startswith('NUS'):
        path = './dataset/NUSWIDE/NUS_pca98.mat'
    elif dataset.startswith('BDGP'):
        path = './dataset/BDGP/BDGP_pca98.mat'
    else:
        raise "unknown dataset"
    data = loadmat(path)
    data_list = []
    for i in range(n_view):
        x = data[f"X{i+1}"]
        data_list.append(torch.tensor(x, dtype=dtype))
    lbl = data['Label'].T
    lbl = lbl[0]
    if min(np.unique(lbl)) == 1:
        lbl = lbl - 1
    N_sam_fea = []
    for i in range(n_view):
        N_sam_fea.append(data_list[i].shape[1])
    class_num = lbl.max() + 1
    N_sample = data_list[0].shape[0]
    return data_list, lbl, N_sample, N_sam_fea, class_num

def make_tensor(data_xy):
    """converts the input to numpy arrays"""
    data_x, data_y = data_xy
    data_x = torch.tensor(data_x)
    data_y = np.asarray(data_y, dtype='int32')
    return data_x, data_y

def load_pickle(f):
    """
    loads and returns the content of a pickled file
    it handles the inconsistencies between the pickle packages available in Python 2 and 3
    """
    try:
        import cPickle as thepickle
    except ImportError:
        import _pickle as thepickle

    try:
        ret = thepickle.load(f, encoding='latin1')
    except TypeError:
        ret = thepickle.load(f)

    return ret


def clustering(lbl, x1, x2=None):
    if x2 is not None:
        x = np.concatenate((x1, x2), axis=1)
    else:
        x = x1
    ulbl = np.unique(lbl)
    n_class = ulbl.shape[0]
    kmeans = KMeans(n_clusters=n_class, random_state=0).fit(x)
    clbl = kmeans.labels_
    import metrics_2
    nmi_score = metrics.normalized_mutual_info_score(labels_true=lbl, labels_pred=clbl)
    pur_score = metrics_2.purity_score(y_true=lbl, y_pred=clbl)#purity(labels_true=lbl, labels_pred=clbl)
    acc_score = accuracy(labels_true=lbl, labels_pred=clbl)
    return nmi_score, pur_score, acc_score

def cluster_eval(y_true, y_pred):
    import metrics_2
    nmi_score = metrics.normalized_mutual_info_score(labels_true=y_true, labels_pred=y_pred)
    pur_score = metrics_2.purity_score(y_true=y_true, y_pred=y_pred)
    acc_score = accuracy(labels_true=y_true, labels_pred=y_pred)
    return nmi_score, pur_score, acc_score

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

















