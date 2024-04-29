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
import matplotlib.gridspec as gridspec
import numpy as np
import torch

import warnings
from torch.serialization import SourceChangeWarning
warnings.filterwarnings("ignore", category=SourceChangeWarning)

def my_corr(output, target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr)

def get_preds_and_targets(odenet, data_handler, method):
    data_pw, t_pw, target_pw = data_handler.get_true_mu_set_pairwise(batch_type = "single")
    print(target_pw.shape)
    #odenet.eval()
    with torch.no_grad():
        predictions_pw = torch.zeros(data_pw.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t_pw, data_pw)):
            predictions_pw[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] 
        corr_explained_pw = my_corr(predictions_pw, target_pw)
        
    return[predictions_pw, target_pw, corr_explained_pw]

if __name__ == "__main__":

    sys.setrecursionlimit(3000)
    
    output_root_dir = 'output/just_plots/'
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50, 'yeast':120}
    models = ["phoenix"]
    datasets = ["yeast"]
    #noises = [0,0.025, 0.05, 0.1]
    
    gene_name_yeast = csv.DictReader(open('/home/ubuntu/phoenix/all_manuscript_models/gene_names_yeast.csv', 'r'))
    gene_name_list_yeast = []
    for line in gene_name_yeast:
        gene_name_list_yeast.append(line)
    
    

    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_rebuttal":"PHOENIX",
                    "phoenix_noprior_rebuttal" :"Unregularized PHOENIX (no prior)"} 
    gene_to_plot_dict = {"yeast": [ 1025, 1701 ]}  # #3350, 2229, 495
    colors = ['green','darkorange', 'darkorchid', 'brown']
    
    leg_yeast = [Patch(facecolor=this_col, edgecolor='black',
                         label= gene_name_list_yeast[this_gene]['name'].replace("_input","",1)) for this_col,this_gene in zip(colors, gene_to_plot_dict['yeast'])]
    
    leg_general_info = [ Line2D([0], [0], label='observed trajectory',
                          linestyle = '-', marker = 'o',  markerfacecolor = 'black', color = 'black'),
                            Line2D([0], [0], label='PHOENIX prediction',
                          linestyle = 'dashed', color='black')]
    
#Plotting setup
    #plt.xticks(fontsize=10)
    #plt.yticks(fontsize=10)

    fig_yeast_res = plt.figure(figsize=(8, 8),constrained_layout=True)
    gs = fig_yeast_res.add_gridspec(2, 2)
    
    border_width = 1.5
    tick_lab_size = 12
    ax_lab_size = 15
    
    #plt.grid(True)
    

    print("......")
    this_data = "yeast"
    dir_yeast = '/home/ubuntu/phoenix/pramila_yeast_data/clean_data/pramila_3551genes_1VALsample_24T.csv'
    data_handler_yeast = DataHandler.fromcsv(dir_yeast, "cpu", val_split = 1, normalize=False, 
                                            batch_type="trajectory", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)
    this_data_handler = data_handler_yeast
    this_neurons = neuron_dict[this_data]
    genes = gene_to_plot_dict[this_data]

    this_odenet = ODENet("cpu", this_data_handler.dim, explicit_time=False, neurons = this_neurons)
    this_odenet.float()
    this_model = "phoenix_rebuttal"
    print("Now on model = {}".format(this_model))

    pretrained_model_file = '/home/ubuntu/phoenix/all_manuscript_models/{}/{}/best_val_model.pt'.format(this_data, this_model)
    this_odenet.load(pretrained_model_file)
        
    trajectories, all_plotted_samples, extrap_timepoints = this_data_handler.calculate_trajectory(this_odenet, 'dopri5', num_val_trajs = 1, yeast = True, time_span = (0, 200))
    times = this_data_handler.time_np
    data_np_to_plot = [this_data_handler.data_np[i] for i in all_plotted_samples]
    data_np_0noise_to_plot = [this_data_handler.data_np_0noise[i] for i in all_plotted_samples]

    ax = fig_yeast_res.add_subplot(gs[0, :])
    ax.spines['bottom'].set_linewidth(border_width)
    ax.spines['left'].set_linewidth(border_width)
    ax.spines['top'].set_linewidth(border_width)
    ax.spines['right'].set_linewidth(border_width)
    ax.cla()

    ax.set_ylim((-3.2, 3.2))
    #ax.set_ylim((-0.8, 0.8))
    ax.set_xlim((-3,200))
    for sample_idx, (approx_traj, traj, true_mean) in enumerate(zip(trajectories, data_np_to_plot, data_np_0noise_to_plot)):
        for gene,this_col in zip(genes, colors):
            with torch.no_grad():
                this_pred_traj = (approx_traj[:,:,gene].numpy().flatten() ) #NORMALIZING to plot! 
                ax.plot(extrap_timepoints, this_pred_traj,
                    color = this_col, linestyle = "--", lw=2, label = "prediction") #times[sample_idx].flatten()[0:] 
                
                noisy_traj =   (traj[:,:,gene].flatten() )
                observed_times = times[sample_idx].flatten()
                ax.plot(observed_times, noisy_traj,    
                color = this_col, lw = 5, linestyle = '-', alpha=0.3)
                ax.plot(observed_times, noisy_traj, 
                linestyle = 'None',
                markerfacecolor = this_col, markeredgecolor = 'black', marker = "o",  alpha=0.8, markersize=7, label = gene)


    ax.tick_params(axis='x', labelsize= tick_lab_size)
    ax.tick_params(axis='y', labelsize= tick_lab_size)

    ax.legend(handles = leg_yeast + leg_general_info, prop={'size': 10}, frameon = False)

    ax.set_ylabel("microarray gene expression", fontsize=ax_lab_size)
    ax.set_xlabel(r'$t$' + " (min)", fontsize=ax_lab_size)
    ax.text(185, -2.7, "(A)", fontsize=16, fontweight='bold', va='bottom', ha='left')
    print("......")

    ax1 = fig_yeast_res.add_subplot(gs[1, 0])   
    ax1.spines['bottom'].set_linewidth(border_width)
    ax1.spines['left'].set_linewidth(border_width)
    ax1.spines['top'].set_linewidth(border_width)
    ax1.spines['right'].set_linewidth(border_width)
    ax1.cla()


    data_handler_yeast_for_corr = DataHandler.fromcsv(dir_yeast, "cpu", val_split = 1, normalize=False, 
                                            batch_type="single", batch_time=100, 
                                            batch_time_frac=0.5,
                                            noise = 0,
                                            img_save_dir = "not needed",
                                            scale_expression = 1)
    
    preds_targets_corr = get_preds_and_targets(this_odenet, data_handler_yeast_for_corr , 'dopri5')
    preds =  (preds_targets_corr[0] ) 
    targets = (preds_targets_corr[1] ) 
    corr =  preds_targets_corr[2].item()
    corr_plot = ax1.plot(targets.squeeze().tolist(), #observed
                         preds.squeeze().tolist(),
                        markerfacecolor = "silver", markeredgecolor = 'black', marker = "o",
                        linestyle = 'None')
    ax1.plot([-3,3], [-3,3], c = "black", linewidth = 1.5, linestyle = "--")
    ax1.text(3, -2, r'$\rho$' + " = {:.4f}".format(corr) ,
            verticalalignment='top', horizontalalignment='right',
            color='black', fontsize=14)
    ax1.set_ylim((-3.2, 3.2))
    ax1.set_xlim((-3.2,3.2))
    ax1.set_ylabel("predicted expression", fontsize=ax_lab_size)
    ax1.set_xlabel('observed expression', fontsize=ax_lab_size)
    ax1.text(-3, 2.5, "(B)", fontsize=16, fontweight='bold', va='bottom', ha='left')
    
    print("......")

    print("Making AUC plots!")
    ax2 = fig_yeast_res.add_subplot(gs[1, 1])
    ax2.spines['bottom'].set_linewidth(border_width)
    ax2.spines['left'].set_linewidth(border_width)
    ax2.spines['top'].set_linewidth(border_width)
    ax2.spines['right'].set_linewidth(border_width)
    ax2.cla()
    
    lambdas = [ 0.2, 0.8, 0.95, 1]
    auc_cols = {0.95: "dodgerblue", 1: "red", 0.2: "navy", 0.8: "skyblue"}
    auc_labs = {
                0.95: r"$\bf{\lambda_{prior}}$" + r"= $\bf{0.05}$ ($\bf{AUC}$ $\bf{0.93}$)", 
                0.80:  r"$\lambda_{prior}$" +"= 0.20 (AUC 0.91)",
                0.20:  r"$\lambda_{prior}$" +"= 0.80 (AUC 0.79)",
                1:  "no prior (AUC 0.54)"}
    
    
    all_auc_vals = {}
    for this_lambda in lambdas:
        if this_lambda not in all_auc_vals:
            all_auc_vals[this_lambda] = {"fpr": [], "tpr": []}
        this_csv = csv.DictReader(open('/home/ubuntu/phoenix/all_manuscript_models/yeast/auc_curves_rebuttal/prior_{}.csv'.format(this_lambda), 'r'))
        for line in this_csv:
            all_auc_vals[this_lambda]["fpr"].append(float(line["fpr"]))
            all_auc_vals[this_lambda]["tpr"].append(float(line["tpr"]))
    #print(all_auc_vals)
    for this_lambda in lambdas:
        ax2.plot( all_auc_vals[this_lambda]["fpr"], all_auc_vals[this_lambda]["tpr"],
                 color = auc_cols[this_lambda], lw=2, linestyle = "-")
    
    ax2.plot([-0.5,1.5], [-0.5,1.5], c = "black", linewidth = 1.5, linestyle = "--")
    
    ax2.set_ylim((-0.05, 1.05))
    ax2.set_xlim((-0.05, 1.05))
    ax2.set_xlabel("FPR", fontsize=ax_lab_size)
    ax2.set_ylabel("TPR", fontsize=ax_lab_size)
    ax2.yaxis.tick_right()
    
    leg_auc = [ Line2D([0], [0], label=auc_labs[this_lambda], lw = 3,
                          linestyle = '-',  color =auc_cols[this_lambda]) for this_lambda in lambdas]    
    ax2.legend(handles = leg_auc, prop={'size': 10}, frameon = False)
    ax2.text(0, 0.93, "(C)", fontsize=16, fontweight='bold', va='bottom', ha='left')
    

    fig_yeast_res.savefig('{}/manuscript_fig_yeast_extra_rebuttal.png'.format(output_root_dir), bbox_inches='tight')    