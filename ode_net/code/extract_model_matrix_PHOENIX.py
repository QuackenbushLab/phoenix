# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
from tqdm import tqdm
from math import ceil
from time import perf_counter, process_time

import torch
import torch.optim as optim

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *
import matplotlib.pyplot as plt

#torch.set_num_threads(4) #CHANGE THIS!

def make_mask(X):
    triu = np.triu(X)
    tril = np.tril(X)
    triuT = triu.T
    trilT = tril.T
    masku = abs(triu) > abs(trilT)
    maskl = abs(tril) > abs(triuT)
    main_mask = ~(masku | maskl)
    X[main_mask] = 0



sums_model = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_sums.pt')
prods_model = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_prods.pt')
alpha_comb = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_alpha_comb.pt')
gene_mult = torch.load('/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model_gene_multipliers.pt')

Wo_sums = np.transpose(sums_model.linear_out.weight.detach().numpy())
Bo_sums = np.transpose(sums_model.linear_out.bias.detach().numpy())
Wo_prods = np.transpose(prods_model.linear_out.weight.detach().numpy())
Bo_prods = np.transpose(prods_model.linear_out.bias.detach().numpy())
alpha_comb = np.transpose(alpha_comb.linear_out.weight.detach().numpy())
gene_mult = np.transpose(torch.relu(gene_mult.detach()).numpy())


num_features = alpha_comb.shape[0]
effects_mat = np.matmul(Wo_sums,alpha_comb[0:num_features//2]) + np.matmul(Wo_prods,alpha_comb[num_features//2:num_features])

num_cols = effects_mat.shape[1]
effects_mat = effects_mat * np.transpose(gene_mult)

np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", effects_mat, delimiter=",")

