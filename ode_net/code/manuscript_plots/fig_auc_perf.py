import os
import numpy as np
import csv 

from visualization_inte import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np

def my_relu_func(x):
    if x > 0:
        return x
    return 0

if __name__ == "__main__":

    save_file_name = "just_plots"
    output_root_dir = '{}/{}/'.format('output', save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    data_letters = {"sim350": "A", "sim690": "B"}
    num_gene_dict = {"sim350": 350, "sim690": 690}
    models = ["phoenix", "phoenix_noprior", "ootb_tanh"]
    datasets = ["sim350", "sim690"]
    noises = [0, 0.025, 0.05, 0.1] #
    perf_info = {}
    metrics = ['opt_TP', 'causal_AUC','opt_TN', 'sparse_out_deg_cor', 'opt_avg_degree' ]
    metric_labels = {'opt_TP':r'$\rm{TPR}_{\max}$', 'causal_AUC':'AUC','opt_TN':r'$\rm{TNR}_{\max}$',
                     'sparse_out_deg_cor':r'$\rho_{\rm{out}}$', 'opt_avg_degree': r'$\mathcal{C}_{\max}$'}
    model_colors = {"phoenix":"dodgerblue", "phoenix_noprior" :"red", "ootb_tanh" : "saddlebrown"} 
    model_alphas = {"phoenix":0.20, "phoenix_noprior" :0.20, "ootb_tanh" : 0.35} 
    
    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_noprior" :"Unregularized PHOENIX (no prior)",
                    "ootb_tanh" : "Out-of-the-box NeuralODE"} 
    
    leg_general_info = [Patch(facecolor=model_colors[this_model], edgecolor= "black", alpha = 0.8,
                         label= model_labels[this_model]) for this_model in models]
                     
    for this_data in datasets:
        if this_data not in perf_info:
            perf_info[this_data] = {}
        for this_model in models:
            if this_model not in perf_info[this_data]:
                perf_info[this_data][this_model] = {}
            for this_noise in noises: 
                if this_noise not in perf_info[this_data][this_model]:
                    perf_info[this_data][this_model][this_noise] = {this_metric: 0 for this_metric in metrics}
    
    perf_csv = csv.DictReader(open('C:/STUDIES/RESEARCH/phoenix/all_manuscript_models/perf_plotting.csv', 'r'))
    for line in perf_csv: 
        if line['model'] in models and float(line['noise']) in noises:
            for this_metric in metrics:
                if this_metric == 'opt_avg_degree':
                    scale_to_start = 0
                    perf_info[line['dataset'].lower()][line['model']][float(line['noise'])][this_metric] = ((1- float(line[this_metric])/num_gene_dict[line['dataset'].lower()])-scale_to_start)/(1- scale_to_start)
                else:
                    perf_info[line['dataset'].lower()][line['model']][float(line['noise'])][this_metric] = float(line[this_metric])

    #Plotting setup
    fig_auc_perfs = plt.figure(figsize=(16,8))
    #plt.grid(True)
    axes_auc_perfs = fig_auc_perfs.subplots(nrows= len(datasets), ncols=len(noises), 
    sharex=False, sharey=True, 
    subplot_kw={'frameon':True, 'projection':'polar'})
    #fig_auc_perfs.subplots_adjust(hspace=0.01, wspace=0.01)
    border_width = 1.5
    tick_lab_size = 11
    ax_lab_size = 15
    plot_metrics = [*metrics, metrics[0]]
    label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(plot_metrics))
    plt.rcParams['axes.facecolor'] = 'silver'

    print("......")
    
    for this_data in datasets:
        for this_model in models:
            for this_noise in noises:
                print("Now on data = {}, noise = {}".format(this_data, this_noise))
                radar_data = [my_relu_func(perf_info[this_data][this_model][this_noise][this_metric]) for this_metric in plot_metrics]
                
                row_num = datasets.index(this_data)
                this_row_plots = axes_auc_perfs[row_num]
                col_num = noises.index(this_noise)
                #col_num = datasets.index(this_data)*len(noises) + noises.index(this_noise)
                ax = this_row_plots[col_num]
                ax.tick_params(axis='x', labelsize= tick_lab_size)
                ax.grid(visible = True, which = "both", axis = "both", color = "black", 
                        linestyle = "-", alpha = 0.3)
                #ax.tick_params(axis='y', labelsize= tick_lab_size)

                #ax.cla()
                #if not(this_model == "ootb_tanh" and this_data == "sim350" and this_noise in [0.05]):
                ax.plot(label_loc, radar_data, color = model_colors[this_model])
                ax.fill(label_loc, radar_data, facecolor= model_colors[this_model], alpha=model_alphas[this_model])
                ax.set_thetagrids(angles = np.degrees(label_loc), 
                                    labels=[metric_labels[this_metric] for this_metric in plot_metrics])
                data_letter = data_letters[this_data]
                if col_num == 0:
                    ax.set_ylabel('({}) {}'.format(data_letter, this_data.upper()), fontsize=ax_lab_size + 5)
                if row_num == 0:
                    this_title = "Noise level = {:.0%}".format(this_noise/0.5) 
                    ax.set_title(this_title, fontsize = ax_lab_size, pad = 15)    
    
    fig_auc_perfs.legend(handles = leg_general_info, loc='lower center', prop={'size': 15}, ncol = 4)
    fig_auc_perfs.savefig('{}/manuscript_fig_auc_perf_rebuttal.png'.format(output_root_dir), bbox_inches='tight')
    