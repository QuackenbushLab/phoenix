import os
import numpy as np
import pandas as pd
import argparse
import pyreadr
import scanpy as sc
import anndata
import sklearn
import umap

import annoy
import torch
from torch import nn, optim
from collections import OrderedDict
from types import SimpleNamespace


def prescient_read_data(data_path, meta_path, out_dir, tp_col, celltype_col, val_set, train_set, noise_sd =0):
    """
    - Load csv preprocessed with scanpy or Seurat.
    - Must be a csv with format n_cells x n_genes with normalized (not scaled!) expression.
    - Must have meta data with n_cells x n_metadata and include timepoints and assigned cell type labels.

    Inputs:
    -------
    path: path to csv or rds file of processed scRNA-seq dataset.
    meta: path to metadata csv.
    """
    ext = os.path.splitext(data_path)[1]
    # load in expression dataframe
    if ext == ".csv" or ext == ".txt" or ext == ".tsv":
        if meta_path == None:
            raise ValueError("Must provide path to metadata with timepoint and ")
        expr = pd.read_csv(data_path, index_col=0)
        meta = pd.read_csv(meta_path)
        genes = expr.columns
        expr = expr.to_numpy()
        tps = meta[tp_col].values.astype(int)
        celltype = meta[celltype_col].values

    # todo: implement Scanpy anndata functionality
    if ext == ".h5ad":
        raise NotImplementedError

    # todo: implement Seurat object functionality
    if ext == ".rds":
        raise NotImplementedError

    # transformations
    scaler = sklearn.preprocessing.StandardScaler()
    num_pcs = 50
    num_neighbors_umap = 10
    pca = sklearn.decomposition.PCA(n_components = num_pcs)
    um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = num_neighbors_umap)

    #x = scaler.fit_transform(expr)
    x = expr #no standardization
    xp = pca.fit_transform(x)
    xu = um.fit_transform(xp)

    y = list(np.sort(np.unique(tps)))

    #make noisy training trajectories
    x_train = [torch.from_numpy(x[(meta[tp_col] == d),:][train_set, :] ).float() + noise_sd*torch.randn(size = (len(train_set), len(genes))) for d in y]
    #make noise-free val trajectories 
    x_val = [torch.from_numpy(x[(meta[tp_col] == d),:][val_set, :] ).float() + 0*torch.randn(size = (len(val_set), len(genes))) for d in y]
    
    xu_ = [torch.from_numpy(xu[(meta[tp_col] == d),:]).float() for d in y]

    #w_pt = torch.load(growth_path)
    #w = w_pt["w"]
    w = [np.ones(150) for i in range(5)] #same weights!

    # write as a torch object
    
    ret_dict = {
     "data": expr,
     "genes": genes,
     "celltype": celltype,
     "tps": tps,
     "x":x_train,
     "xp":x_train, #xp_
     "x_val": x_val,
     "xu": xu_,
     "y": y,
     "pca": pca,
     "um":um,
     "w":w,
     "out_dir": out_dir
     }

    return(ret_dict)

class AutoGenerator(nn.Module):

    def __init__(self, config):
        super(AutoGenerator, self).__init__()

        self.x_dim = config.x_dim
        self.k_dim = config.k_dim
        self.layers = config.layers

        self.activation = config.activation
        if self.activation == 'relu':
            self.act = nn.LeakyReLU
        elif self.activation == 'softplus':
            self.act = nn.Softplus
        elif self.activation == 'intrelu': # broken, wip
            raise NotImplementedError
        elif self.activation == 'none':
            self.act = None
        else:
            raise NotImplementedError

        self.net_ = []
        for i in range(self.layers):
            # add linear layer
            if i == 0:
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.x_dim, self.k_dim)))
            else:
                self.net_.append(('linear{}'.format(i+1), nn.Linear(self.k_dim, self.k_dim)))
            # add activation
            if self.activation == 'intrelu':
                raise NotImplementedError
            elif self.activation == 'none':
                pass
            else:
                self.net_.append(('{}{}'.format(self.activation, i+1), self.act()))
        self.net_.append(('linear', nn.Linear(self.k_dim, 1, bias = False)))
        self.net_ = OrderedDict(self.net_)
        self.net = nn.Sequential(self.net_)

        net_params = list(self.net.parameters())
        net_params[-1].data = torch.zeros(net_params[-1].data.shape) # initialize

    def _step(self, x, dt, z):
        sqrtdt = np.sqrt(dt)
        return x + self._drift(x) * dt + z * sqrtdt

    def _pot(self, x):
        return self.net(x)

    def _drift(self, x):
        x_ = x.requires_grad_()
        pot = self._pot(x_)

        drift = torch.autograd.grad(pot, x_, torch.ones_like(pot),
            create_graph = True)[0]
        return drift
   


