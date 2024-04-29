import sys
import os
import numpy as np
import csv 
import math 

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def make_mask(X):
    triu = np.triu(X)
    tril = np.tril(X)
    triuT = triu.T
    trilT = tril.T
    masku = abs(triu) > abs(trilT)
    maskl = abs(tril) > abs(triuT)
    main_mask = ~(masku | maskl)
    X[main_mask] = 0

if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format('val_config_inte.cfg'))
    settings = read_arguments_from_file('val_config_inte.cfg')
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    models = ["phoenix_noprior", "phoenix"]
    datasets = ["sim350"]
    noises = [0,0.025, 0.1, 0.4]
    
    
    datahandler_dim = {"sim350": 350}
    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_noprior" :"Unregularized PHOENIX (no prior)"} 
    
    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")    
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_heat_sparse = plt.figure(figsize=(10,18)) # tight_layout=True
    axes_heat_sparse = fig_heat_sparse.subplots(ncols= len(models), nrows=len(noises), 
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_heat_sparse.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 15
    color_mult = 0.26 #0.25
    
    plt.grid(True)
    
    print("......")
    
    for this_data in datasets:
        #this_data_handler = datahandler_dict[this_data]
        this_neurons = neuron_dict[this_data]
        this_odenet = ODENet("cpu", datahandler_dim[this_data], explicit_time=False, neurons = this_neurons)
        this_odenet.float()
        for this_noise in noises:
            noise_string = "noise_{}".format(this_noise)
            for this_model in models:
                print("Now on model = {}, noise = {}".format(this_model, this_noise))
                
                row_num = noises.index(this_noise)
                this_row_plots = axes_heat_sparse[row_num]
                col_num = models.index(this_model)
                ax = this_row_plots[col_num]
                ax.spines['bottom'].set_linewidth(border_width)
                ax.spines['left'].set_linewidth(border_width)
                ax.spines['top'].set_linewidth(border_width)
                ax.spines['right'].set_linewidth(border_width)
                ax.cla()

                pretrained_model_file = 'C:/STUDIES/RESEARCH/phoenix/all_manuscript_models/{}/{}/{}/best_val_model.pt'.format(this_data, this_model, noise_string)
                this_odenet.load(pretrained_model_file)
                Wo_sums = np.transpose(this_odenet.net_sums.linear_out.weight.detach().numpy())
                Wo_prods = np.transpose(this_odenet.net_prods.linear_out.weight.detach().numpy())
                alpha_comb = np.transpose(this_odenet.net_alpha_combine.linear_out.weight.detach().numpy())
                gene_mult = np.transpose(torch.relu(this_odenet.gene_multipliers.detach()).numpy()) 

                y, x = np.meshgrid(np.linspace(1, datahandler_dim[this_data], datahandler_dim[this_data]), np.linspace(1, datahandler_dim[this_data], datahandler_dim[this_data]))
                z = np.matmul(Wo_sums, alpha_comb[0:this_neurons,]) + np.matmul(Wo_prods, alpha_comb[this_neurons:(2*this_neurons),])    
                z = z* gene_mult.reshape(1, -1) 
                row_sums =  np.abs(z).sum(axis=1)
                z = z / row_sums[:, np.newaxis]

                z_min, z_max = color_mult*-np.abs(z).max(), color_mult*np.abs(z).max()
                c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max) 
                ax.axis([x.min(), x.max(), y.min(), y.max()]) 
                
                if row_num == 0 and col_num == 0:
                    fig_heat_sparse.canvas.draw()
                    labels_y = [item.get_text() for item in ax.get_yticklabels()]
                    labels_y_mod = [(r"$g'$"+item).translate(SUB) for item in labels_y]
                    labels_x = [item.get_text() for item in ax.get_xticklabels()]
                    labels_x_mod = [(r'$g$'+item).translate(SUB) for item in labels_x]
                
                ax.set_xticklabels(labels_x_mod)
                ax.set_yticklabels(labels_y_mod)
                ax.tick_params(axis='x', labelsize= tick_lab_size)
                ax.tick_params(axis='y', labelsize= tick_lab_size)
                    
                if row_num == 0:
                    ax.set_title(model_labels[this_model], fontsize=ax_lab_size, pad = 10)
                if col_num == 0:
                    ax.set_ylabel("Noise level = {:.0%}".format(this_noise/0.5), fontsize = ax_lab_size) 
                 
    cbar =  fig_heat_sparse.colorbar(c, ax=axes_heat_sparse.ravel().tolist(), 
                                        shrink=0.95, orientation = "horizontal", pad = 0.05)
    cbar.set_ticks([0, 0.006, -0.006])
    cbar.set_ticklabels(['None', 'Activating', 'Repressive'])
    cbar.ax.tick_params(labelsize = tick_lab_size+3) 
    cbar.set_label(r'$\widetilde{D_{ij}}$= '+'Estimated effect of '+ r'$g_j$'+ ' on ' +r"$\frac{dg_i}{dt}$" +' in SIM350', size = ax_lab_size)
    cbar.outline.set_linewidth(2)

    
    fig_heat_sparse.savefig('{}/manuscript_fig_heat_sparse_rebuttal.png'.format(output_root_dir), bbox_inches='tight')
    