import os
import numpy as np
import csv 

from visualization_inte import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon
import numpy as np

if __name__ == "__main__":

    save_file_name = "just_plots"
    output_root_dir = '{}/{}/'.format('output', save_file_name)
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    
    neuron_dict = {"sim350": 40, "sim690": 50}
    num_gene_dict = {"sim350": 350, "sim690": 690}
    models = ["phoenix","phoenix_noprior", "ootb_tanh",
                "ootb_relu","ootb_sigmoid"]
    datasets = ["sim350", "sim690"]
    noises = [0, 0.1, 0.4]
    perf_info = {}
    metrics = ['true_val_MSE_1', 'true_val_MSE_2', 'true_val_MSE_3', 'true_val_MSE_4','true_val_MSE_5']
    model_colors = {"phoenix":"dodgerblue", "phoenix_noprior" :"red", 
                    "ootb_tanh" : "saddlebrown", "ootb_relu" : "sandybrown", "ootb_sigmoid" : "peachpuff"} 
    model_labels = {"phoenix":"PHOENIX", 
                    "phoenix_noprior" :"Unregularized PHOENIX",
                    "ootb_tanh" : "OOTB (tanh)",
                    "ootb_relu" : "OOTB (ReLU)",
                    "ootb_sigmoid" : "OOTB (sigmoid)"} 

    model_hatch = {"phoenix":"", "phoenix_noprior" :"", 
                    "ootb_tanh" : "//", "ootb_relu" : "..", "ootb_sigmoid" : "xx"}                
    
    leg_general_info = [Patch(facecolor=model_colors[this_model], edgecolor= "black", 
                            alpha = 0.7, hatch = model_hatch[this_model],
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
                perf_info[line['dataset'].lower()][line['model']][float(line['noise'])][this_metric] = float(line[this_metric])

    #Plotting setup
    fig_ootb_mse = plt.figure(figsize=(18,11))
    #plt.grid(visible = True)
    axes_ootb_mse = fig_ootb_mse.subplots(ncols= len(datasets),nrows = 1,  
    sharex=False, sharey=False, 
    subplot_kw={'frameon':True})
    #fig_ootb_mse.subplots_adjust(hspace=0.0, wspace=0.0)
    border_width = 1.5
    tick_lab_size = 11
    ax_lab_size = 20
    ind = np.arange(len(noises))  # the x locations for the groups
    height= 0.15  # the width of the bars
    deltas = [-2, -1, 0, 1, 2]
    print("......")
    
    for this_data in datasets:
        print("Now on data = {}".format(this_data))
        for this_model in models:
            col_num = datasets.index(this_data)
            ax = axes_ootb_mse[col_num]
            ax.spines['bottom'].set_linewidth(border_width)
            ax.spines['left'].set_linewidth(border_width)
            ax.spines['top'].set_linewidth(border_width)
            ax.spines['right'].set_linewidth(border_width)
            ax.set_xlim((6.5*10**-4,6.2*10**-3))
            ax.set_yticks(ind)
            ax.set_yticklabels(['0% noise', '20% noise', '80% noise'], rotation=90, va = "center")
            ax.tick_params(axis='x', labelsize= tick_lab_size)
            ax.tick_params(axis='y', labelsize= 17)
            
            this_delta = deltas[models.index(this_model)] 
            this_model_mses =  [np.mean([perf_info[this_data][this_model][this_noise][true_val_col] 
                                            for true_val_col in metrics]) for this_noise in noises]
            this_model_stdev =  [np.std([perf_info[this_data][this_model][this_noise][true_val_col] 
                                            for true_val_col in metrics]) for this_noise in noises]                                
            ax.barh(ind + height*this_delta, this_model_mses, height = height, 
                    xerr = this_model_stdev, capsize = 5,
                    color = model_colors[this_model], alpha = 0.7,  edgecolor = "black", 
                    linewidth = 1.5, align = 'center', hatch = model_hatch[this_model])
            ax.set_xscale("log")
            ax.grid(visible = True, which = "both", axis = "x", color = "black", 
            linestyle = "--", alpha = 0.3)
            
        ax.set_xlabel('validation MSE ({})'.format(this_data.upper()), fontsize=ax_lab_size+5)
        if col_num == 0:
            ax.set_ylabel('Noise level', fontsize=ax_lab_size+5)
    
    fig_ootb_mse.legend(handles = leg_general_info, loc='upper center', prop={'size': 16}, 
                        ncol = 3,  handleheight=1.5, frameon = False)
    fig_ootb_mse.savefig('{}/manuscript_fig_ootb_mse_rebuttal.png'.format(output_root_dir), bbox_inches='tight')
    