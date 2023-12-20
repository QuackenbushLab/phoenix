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

def get_true_val_velocities_new(odenet, data_handler, data_handler_velo,  method, batch_type, noise_for_training = 0, scale_factor_for_counts = 1, breast = False, yeast_test_handler = None):
    print("NOISE = {}".format(noise_for_training))
    data_pw, t_pw, data_target = data_handler.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    true_velo_pw, _unused1, _unused2 = data_handler_velo.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)

    #data_pw = torch.cat((data_pw, data_pw_test))
    #t_pw = torch.cat((t_pw, t_pw_test))
    #data_target = torch.cat((data_target, target_pw_test))
    
    data_pw = data_pw  * scale_factor_for_counts
    data_target = data_target  * scale_factor_for_counts
    true_velo_pw = true_velo_pw * scale_factor_for_counts

    if breast:
        dyn_test_set = range(92, 99) #np.random.choice(range(185), 8, replace=False)
        dyn_val_set = np.random.choice(np.setdiff1d(range(185), dyn_test_set), 7, replace=False)
        dyn_train_set = np.setdiff1d(range(185), np.concatenate((dyn_val_set, dyn_test_set)))
    
    else:
        if yeast_test_handler is None:    
            dyn_val_set = np.random.choice(range(640), 40, replace=False)
            dyn_test_set = np.random.choice(np.setdiff1d(range(640), dyn_val_set), 40, replace=False)
            dyn_train_set = np.setdiff1d(range(640), np.concatenate((dyn_val_set, dyn_test_set)))
        else: #doing yeast
            dyn_val_set = np.random.choice(range(23), 3, replace=False)
            dyn_train_set = np.setdiff1d(range(23), dyn_val_set)


    #dyn_val_set = np.arange(0, 40) 
    #dyn_test_set = np.arange(40, 80) 
    #dyn_train_set = np.arange(80, 640)

    #make noisy training data
    data_pw_train = data_pw[dyn_train_set, :, :]
    data_pw_train = data_pw_train +  noise_for_training*torch.randn(size = data_pw_train.shape)
    data_target_train = data_target[dyn_train_set, :, :]  
    data_target_train = data_target_train +  noise_for_training*torch.randn(size = data_target_train.shape)
    t_pw_train =t_pw[dyn_train_set, : ]
    true_velo_train = true_velo_pw[dyn_train_set, :, :] + 0
    print("*******NOISELESS velos for training!!!************")
    
    #make noise-free validation data
    data_pw_val = data_pw[dyn_val_set, :, :]  
    data_pw_val = data_pw_val +  noise_for_training*torch.randn(size = data_pw_val.shape)
    data_target_val = data_target[dyn_val_set, :, :]  
    data_target_val = data_target_val +  noise_for_training*torch.randn(size = data_target_val.shape)
    t_pw_val =t_pw[dyn_val_set, : ]
    true_velo_val = true_velo_pw[dyn_val_set, :, :]+ 0

    if yeast_test_handler is None:
        data_pw_test = data_pw[dyn_test_set, :, :]  
        target_pw_test = data_target[dyn_test_set, :, :]  
        t_pw_test =t_pw[dyn_test_set, : ]
    else:
        data_pw_test, t_pw_test, target_pw_test = yeast_test_handler.get_true_mu_set_pairwise(val_only = False, batch_type =  batch_type)
    

    phx_val_set_pred = scale_factor_for_counts* odenet.forward(t_pw_val,data_pw_val/scale_factor_for_counts)
    
    data_pw_train = torch.squeeze(data_pw_train).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw_val = torch.squeeze(data_pw_val).detach().numpy() #convert to NP array for dynamo VF alg
    data_target_train = torch.squeeze(data_target_train).detach().numpy()
    t_pw_train =  torch.squeeze(t_pw_train).detach().numpy()
    
    data_target_val = torch.squeeze(data_target_val).detach().numpy()
    true_velo_train = torch.squeeze(true_velo_train).detach().numpy() #convert to NP array for dynamo VF alg
    t_pw_val =  torch.squeeze(t_pw_val).detach().numpy()
    true_velo_val = torch.squeeze(true_velo_val).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw_test = torch.squeeze(data_pw_test).detach().numpy() #convert to NP array for dynamo VF alg
    target_pw_test = torch.squeeze(target_pw_test).detach().numpy()
    t_pw_test =  torch.squeeze(t_pw_test).detach().numpy()

    phx_val_set_pred = torch.squeeze(phx_val_set_pred).detach().numpy() #convert to NP array for dynamo VF alg
    data_pw = torch.squeeze(data_pw).detach().numpy()
    true_velo_pw = torch.squeeze(true_velo_pw).detach().numpy()
    
    

    return {'x_train': data_pw_train, 'true_velo_x_train': true_velo_train, 
            'x_val': data_pw_val, 'true_velo_x_val': true_velo_val,
            "t_val": t_pw_val, "x_target_val" : data_target_val, 
            'phx_val_set_pred' : phx_val_set_pred,
            'x_full': data_pw, 'true_velo_x_full': true_velo_pw,
            'x_test': data_pw_test, 't_test': t_pw_test,
            'x_target_test': target_pw_test, 'x_target_train': data_target_train,
            't_train' : t_pw_train} 




def pred_traj_given_ode(my_ode_func, X_val, t_val, method = None, breast_test = False):
    all_preds = np.copy(X_val)
    if not breast_test:
        for val_idx in range(len(X_val)):
            #print("predicting val_idx:", val_idx)
            s = solve_ivp(fun = my_ode_func, 
                            t_span = t_val[val_idx,], 
                            y0 = X_val[val_idx,], 
                            t_eval = t_val[val_idx,])
            this_pred = s['y'][:, 1]
            all_preds[val_idx, :] = this_pred
        return(all_preds)
    else:
        init_val = X_val[0]
        all_t = sorted(np.unique(t_val.flatten()))

        s = solve_ivp(fun = my_ode_func, 
                            t_span = (t_val[0][0], t_val[-1][1]), 
                            y0 = init_val, 
                            t_eval =  all_t)
        all_preds = s['y'][:, 1:].transpose()
        return(all_preds)
    



#VELO lambda FUNCTION FOR PHOENIX:
#velo_fun_x = lambda t,x : torch.squeeze(odenet.forward(torch.tensor([99,999]), torch.unsqueeze(torch.from_numpy(x), dim = 0).float())).detach().numpy()
       