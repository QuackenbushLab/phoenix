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
from solve_eq import solve_eq
from visualization_inte import *

#torch.set_num_threads(16) #CHANGE THIS!


def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name =  "desmedt_11165genes_1sample_186T" 
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/{}.csv'.format(clean_name))

args = parser.parse_args()
device = "cpu"
# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    
    data_handler = DataHandler.fromcsv(args.data,device , settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = settings['noise'],
                                        img_save_dir = "NULL",
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])

    
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                    log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
    odenet.float()
    pretrained_model_file = '/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
    odenet.load(pretrained_model_file)
    print(odenet)

    time_pts_to_project = torch.from_numpy(np.arange(0,1,0.1))
    n_random_inputs_per_gene = 60
    all_scores = []
    #Read in the prior matrix
    for this_gene in range(data_handler.dim):
        this_init = 1*(torch.rand(n_random_inputs_per_gene,1,data_handler.dim, device = data_handler.device) - 0.5)
        unpert_out = odeint(odenet,this_init, time_pts_to_project, method= settings['method'])
        this_pert_col =  1*(torch.rand(n_random_inputs_per_gene,device = data_handler.device) - 0.5)
        this_init[:,0, this_gene] = this_pert_col 
        pert_out = odeint(odenet,this_init, time_pts_to_project, method= settings['method']) 
        all_other_genes = [idx for idx in range(data_handler.dim) if idx != this_gene]
        this_gene_score = torch.mean(abs(unpert_out[1:,:,:,all_other_genes] - pert_out[1:,:,:,all_other_genes]))
        all_scores.append(this_gene_score.item())
        print(this_gene)

    print("done, saving now!")
    np.savetxt('/home/ubuntu/neural_ODE/ode_net/code/model_inspect/inferred_influence_11165.csv', 
                    all_scores, delimiter=',') 