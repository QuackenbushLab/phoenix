# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np

import torch
import sklearn
import scipy
import tensorflow as tf
from tensorflow import keras

from scipy.integrate import odeint, solve_ivp
import umap
import pandas as pd

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *


def get_true_val_velocities(odenet, data_handler, data_handler_velo, method, batch_type, noise_for_training = 0):
    data_pw, t_pw, data_target = data_handler.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    true_velo_pw, _unused1, _unused2 = data_handler_velo.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    
    scale_factor_for_counts = 1
    val_split = 0.10

    data_pw = data_pw  * scale_factor_for_counts
    true_velo_pw = true_velo_pw * scale_factor_for_counts

    num_samples = data_pw.shape[0]
    n_dyn_val = int(num_samples*val_split)
    dyn_val_set = np.random.choice(range(num_samples), n_dyn_val, replace=False)
    dyn_train_set = np.setdiff1d(range(num_samples), dyn_val_set)

    #make noisy training data
    data_pw_train = data_pw[dyn_train_set, :, :]
    data_pw_train = data_pw_train +  noise_for_training*torch.randn(size = data_pw_train.shape)
    true_velo_train = true_velo_pw[dyn_train_set, :, :] + 0
    print("*******NOISELESS velos for training!!!************")
    
    #make noise-free validation data
    data_pw_val = data_pw[dyn_val_set, :, :]
    data_target_val = data_target[dyn_val_set, :, :]
    t_pw_val =t_pw[dyn_val_set, : ]
    true_velo_val = true_velo_pw[dyn_val_set, :, :]

    phx_val_set_pred = scale_factor_for_counts* odenet.forward(t_pw_val,data_pw_val/scale_factor_for_counts)
    
    data_pw_train = torch.squeeze(data_pw_train).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw_val = torch.squeeze(data_pw_val).detach().numpy() #convert to NP array for dynamo VF alg
    data_target_val = torch.squeeze(data_target_val).detach().numpy()
    true_velo_train = torch.squeeze(true_velo_train).detach().numpy() #convert to NP array for dynamo VF alg
    t_pw_val =  torch.squeeze(t_pw_val).detach().numpy()
    true_velo_val = torch.squeeze(true_velo_val).detach().numpy() #convert to NP array for dynamo VF alg
    phx_val_set_pred = torch.squeeze(phx_val_set_pred).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw = torch.squeeze(data_pw).detach().numpy()
    true_velo_pw = torch.squeeze(true_velo_pw).detach().numpy()

    return {'x_train': data_pw_train, 'true_velo_x_train': true_velo_train, 
            'x_val': data_pw_val, 'true_velo_x_val': true_velo_val,
            "t_val": t_pw_val, "x_target_val" : data_target_val, 
            'phx_val_set_pred' : phx_val_set_pred,
            'x_full': data_pw, 'true_velo_x_full': true_velo_pw} 


def pred_traj_given_ode(my_ode_func, X_val, t_val, method = None):
    all_preds = np.copy(X_val)
    if method != "rnaode":
        for val_idx in range(len(X_val)):
            print("predicting val_idx:", val_idx)
            s = solve_ivp(fun = my_ode_func, 
                            t_span = t_val[val_idx,], 
                            y0 = X_val[val_idx,], 
                            t_eval = t_val[val_idx,])
            this_pred = s['y'][:, 1]
            all_preds[val_idx, :] = this_pred
        return(all_preds)
    else:
        s = solve_ivp(fun = my_ode_func, 
                            t_span = t_val, 
                            y0 = X_val, 
                            t_eval = t_val)
        return(s)
    


#VELO lambda FUNCTION FOR PHOENIX:
#velo_fun_x = lambda t,x : torch.squeeze(odenet.forward(torch.tensor([99,999]), torch.unsqueeze(torch.from_numpy(x), dim = 0).float())).detach().numpy()
       