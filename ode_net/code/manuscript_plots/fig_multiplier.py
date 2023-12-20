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
import random



import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

if __name__ == "__main__":
    
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format('val_config_inte.cfg'))
    settings = read_arguments_from_file('val_config_inte.cfg')
    save_file_name = "just_plots"

    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    data_letters = ["(A) Model fit with multipliers " + r'$\mathbf{\upsilon}$', "(B) Model fit w/o multipliers " + r'$\mathbf{\upsilon}$']
    models = ["phoenix"]
    datasets = ["sim690", "sim690"] # 
    noises = [0, 0.025, 0.05, 0.1]
    perf_info = {}
    for this_data in datasets:
        if this_data not in perf_info:
            perf_info[this_data] = {}
        for this_model in models:
            if this_model not in perf_info[this_data]:
                perf_info[this_data][this_model] = {}
            for this_noise in noises: 
                if this_noise not in perf_info[this_data][this_model]:
                    perf_info[this_data][this_model][this_noise] = {'true_val_MSE_1':0}
    
    perf_csv = csv.DictReader(open('/home/ubuntu/phoenix/all_manuscript_models/perf_plotting.csv', 'r'))
    for line in perf_csv: 
        if line['model'] in models and float(line['noise']) in noises and line['dataset'].lower() in datasets:
            perf_info[line['dataset'].lower()][line['model']][float(line['noise'])]['true_val_MSE_1'] = float(line['true_val_MSE_1'])

    
    gene_name_690 = csv.DictReader(open('/home/ubuntu/phoenix/all_manuscript_models/gene_names_690.csv', 'r'))
    gene_name_list_690 = []
    for line in gene_name_690:
        gene_name_list_690.append(line)
    
    dir_350 = '/home/ubuntu/phoenix/ground_truth_simulator/clean_data/chalmers_350genes_150samples_earlyT_0bimod_1initvar.csv'
    dir_690 = '/home/ubuntu/phoenix/ground_truth_simulator/clean_data/chalmers_690genes_150samples_earlyT_0bimod_1initvar.csv'

    data_handler_350 = DataHandler.fromcsv(dir_350, "cpu", 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)

    data_handler_690 = DataHandler.fromcsv(dir_690, "cpu", 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)
    
    datahandler_dict = {"sim350": data_handler_350, "sim690": data_handler_690}
        
    #Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)
    fig_traj_split = plt.figure(figsize=(15,10)) # tight_layout=True
    axes_traj_split = fig_traj_split.subplots(nrows= len(datasets), ncols=len(noises), 
    sharex=False, sharey=True, 
    subplot_kw={'frameon':True})
    fig_traj_split.subplots_adjust(hspace=0, wspace=0)
    border_width = 1.5
    tick_lab_size = 11
    ax_lab_size = 15
    gene_to_plot_dict = {"sim350": [9, 45, 33, 49], "sim690": [9, 45, 33, 49]} #, 46, 47,
    colors = ['darkorange','deeppink','darkorchid', 'limegreen']
    leg_350 = [Patch(facecolor=this_col, edgecolor='black',
                         label= gene_name_list_690[this_gene]['x'].replace("_input","",1)) for this_col,this_gene in zip(colors, gene_to_plot_dict['sim350'])]
    leg_690 = [Patch(facecolor=this_col, edgecolor='black',
                         label= this_gene) for this_col,this_gene in zip(colors, gene_to_plot_dict['sim690'])]
    
    leg_general_info = [ Line2D([0], [0], label='observed (noisy) trajectory',
                          linestyle = '-', marker = 'o',  markerfacecolor = 'black', color = 'black'),
                            Line2D([0], [0], label='predicted trajectory',
                          linestyle = 'dashed', color='black')]
    plt.grid(True)
    #plt.margins(x = 0.0, y = 0.0)
    
    print("......")
    row_num = -1
    for this_data in datasets:
        np.random.seed(88)
                            
        row_num +=1 
        this_data_handler = datahandler_dict[this_data]
        this_neurons = neuron_dict[this_data]
        genes = gene_to_plot_dict[this_data]
        for this_model in models:
            this_odenet = ODENet("cpu", this_data_handler.dim, explicit_time=False, neurons = this_neurons)
            this_odenet.float()
                
            for this_noise in noises:
                print("Now on data = {}, noise = {}".format(this_data, this_noise))
                noise_string = "noise_{}".format(this_noise)
                true_val_mse = perf_info[this_data][this_model][this_noise]['true_val_MSE_1']
                #print(true_val_mse)
                pretrained_model_file = '/home/ubuntu/phoenix/all_manuscript_models/{}/{}/{}/best_val_model.pt'.format(this_data, this_model, noise_string)
                this_odenet.load(pretrained_model_file)

                if row_num ==1:
                    print("removing multipliers")
                    pretrained_dict = this_odenet.state_dict()
                    curr_multipliers = pretrained_dict['gene_multipliers']
                    all_ones = torch.ones(curr_multipliers.shape)
                    pretrained_dict['gene_multipliers'] = all_ones
                    this_odenet.load_state_dict(pretrained_dict)
                    #print(this_odenet.gene_multipliers)

                
                trajectories, all_plotted_samples, extrap_timepoints = this_data_handler.calculate_trajectory(this_odenet, 'dopri5', num_val_trajs = 1, 
                                                                                                              fixed_traj_idx = [7], time_span = (0, 10))

                times = this_data_handler.time_np
                data_np_to_plot = [this_data_handler.data_np[i] for i in all_plotted_samples]
                data_np_0noise_to_plot = [this_data_handler.data_np_0noise[i] for i in all_plotted_samples]
                
                #row_num = datasets.index(this_data)
                this_row_plots = axes_traj_split[row_num]
                col_num = noises.index(this_noise)
                ax = this_row_plots[col_num]
                ax.spines['bottom'].set_linewidth(border_width)
                ax.spines['left'].set_linewidth(border_width)
                ax.spines['top'].set_linewidth(border_width)
                ax.spines['right'].set_linewidth(border_width)
                ax.tick_params(axis='x', labelsize= tick_lab_size)
                ax.tick_params(axis='y', labelsize= tick_lab_size)

                ax.cla()
                ax.set_xlim((-0.5, 10.5))
                ax.set_ylim((-0.2,1.2))
                for sample_idx, (approx_traj, traj, true_mean) in enumerate(zip(trajectories, data_np_to_plot, data_np_0noise_to_plot)):
                    for gene,this_col in zip(genes, colors):
                        with torch.no_grad():
                            this_pred_traj = approx_traj[:,:,gene].numpy().flatten()
                            ax.plot(extrap_timepoints, this_pred_traj,
                             color = this_col, linestyle = "--", lw=2, label = "prediction") #times[sample_idx].flatten()[0:] 
                            
                            noisy_traj =  traj[:,:,gene].flatten()
                            noisy_traj[1:5] += np.random.normal(0,this_noise,4)
                            observed_times = times[sample_idx].flatten()
                            ax.plot(observed_times, noisy_traj,    
                            color = this_col, lw = 5, linestyle = '-', alpha=0.3)
                            ax.plot(observed_times, noisy_traj, 
                            linestyle = 'None',
                            markerfacecolor = this_col, markeredgecolor = 'black', marker = "o",  alpha=0.8, markersize=7, label = gene)
                            
                #ax.text(10, 1.15, r'$\log_{10}$' +'(test MSE) = {:.2f}'.format(math.log10(true_val_mse)),
                #    verticalalignment='top', horizontalalignment='right',
                   # transform=ax.transAxes,
                 #   color='black', fontsize=12)
                ax.set_xlabel(r'$t$', fontsize=ax_lab_size)
                data_letter = data_letters[row_num]
                if col_num == 0:
                    ax.set_ylabel('{}\n\ngene expression'.format(data_letter, this_data.upper()), fontsize=ax_lab_size)
                if row_num == 0:
                    ax.set_title("Noise level = {:.0%}".format(this_noise/0.5), fontsize = ax_lab_size)    
                
    temp_leg = fig_traj_split.legend(handles = leg_350 + leg_general_info, 
                                        loc='lower center', prop={'size': 15}, ncol = 6)
    #fig_traj_split.legend(handles = leg_690, loc='lower right', prop={'size': 15})
    #fig_traj_split.add_artist(temp_leg)
    #fig_traj_split.tight_layout()
    fig_traj_split.savefig('{}/manuscript_fig_multiplier_rebuttal.png'.format(output_root_dir), bbox_inches='tight')
    