# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np
import sklearn
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
from visualization_compete import *

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import BaseDecisionTree

from GRN_rnaode import *
from helper_true_velo import *

def get_true_val_set_r2(predictions, target, img_save_dir, N_list):
    pred_perf_init_val_based_list =[get_pred_perf_across_top_N_genes(my_pred_tensor = predictions, my_target_tensor = target, N = n) for n in N_list]
    #true_val_mse_init_val_based = np.mean((predictions - target)**2)

    '''
    plt.figure()
    plt.plot(N_list,var_explained_init_val_based, c = "blue", label = "Predict entire traj based on $x(t_0)$")
    plt.ylabel('Variance explained of $N$-most variable genes in test set')
    plt.xlabel('$N$')
    plt.ylim(0, 1.02)
    plt.title('Predictive performance on BRCA test set')
    plt.legend(loc='lower right')
    plt.savefig("{}/rnaode_test_set_R2_by_N.png".format(img_save_dir))
    '''
    return pred_perf_init_val_based_list    

def get_pred_perf_across_top_N_genes(my_target_tensor, my_pred_tensor,  N):
    variance = torch.var(torch.tensor(my_target_tensor), dim=0)
    # Flatten the variance tensor
    flattened_variance = variance.view(-1)
    # Find the indices of the top 5 genes with highest variance
    topN_indices = torch.argsort(flattened_variance, descending=True)[:N]
    subset_target = my_target_tensor[:,topN_indices]
    subset_pred = my_pred_tensor[:,topN_indices]
    r2 =  my_r_squared(subset_pred, subset_target)
    mse = np.mean((subset_pred - subset_target)**2)
    perc_error =  np.mean(np.abs(subset_pred - subset_target)/np.abs(subset_target))

    return([r2, mse, perc_error])



def my_r_squared(output, target):
    x = torch.tensor(output)
    y = torch.tensor(target)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr**2)


def BUILD_MODEL(counts, velocity, genes=None, tfs=None, method='rf', n_estimators=10, max_depth=10, lasso_alpha=1, train_size=0.999):
    '''
    v = f(x, a). return fitted function f
    method: 'rf'(random forest) / 'lasso' / 'linear'.
    '''
    if genes is None:
        genes = np.array([True] * counts.shape[1])
    if tfs is None:
        tfs = genes
    x, _unused1, y, _unused2 = train_test_split(counts[:, tfs], velocity[:, genes], 
                                          test_size=1-train_size, random_state=42)

    # Build model
    if method == 'lasso':
        model = linear_model.Lasso(alpha=lasso_alpha)
    elif method == 'linear':
        model = linear_model.LinearRegression(n_jobs=-1)
    elif method == 'rf':
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=9, n_jobs=-1)  
    model = model.fit(x, y)    
    
    train_score = (model.score(x, y))**0.5
    
    print('Fitted model | Train Corr: {:.4f}'.format(train_score))
    return model, train_score


def _build_save_file_name(save_path, epochs):
    return 'rnaode_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_breast.cfg')
clean_name =  "desmedt_11165genes_1sample_186T" 
clean_name_velo =  "desmedt_11165genes_1sample_186T_DERIVATIVES"  
parser.add_argument('--data', type=str, default='/home/ubuntu/phoenix/breast_cancer_data/clean_data/{}.csv'.format(clean_name))
parser.add_argument('--velo_data', type=str, default='/home/ubuntu/phoenix/breast_cancer_data/clean_data/{}.csv'.format(clean_name_velo))

args = parser.parse_args()

# Main function
if __name__ == "__main__":
    print('Setting recursion limit to 3000')
    sys.setrecursionlimit(3000)
    print('Loading settings from file {}'.format(args.settings))
    settings = read_arguments_from_file(args.settings)
    cleaned_file_name = clean_name
    save_file_name = _build_save_file_name(cleaned_file_name, settings['epochs'])

    if settings['debug']:
        print("********************IN DEBUG MODE!********************")
        save_file_name= '(DEBUG)' + save_file_name
    output_root_dir = '{}/{}/'.format(settings['output_dir'], save_file_name)

    img_save_dir = '{}img/'.format(output_root_dir)
    interm_models_save_dir = '{}interm_models/'.format(output_root_dir)
    #intermediate_models_dir = '{}intermediate_models/'.format(output_root_dir)

    # Use GPU if available
    if not settings['cpu']:
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        print("Trying to run on GPU -- cuda available: " + str(torch.cuda.is_available()))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Running on", device)
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        print("Running on CPU")
        device = 'cpu'
    
    data_handler = DataHandler.fromcsv(args.data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = 0,
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])
    
    data_handler_velo = DataHandler.fromcsv(args.velo_data, device, settings['val_split'], normalize=settings['normalize_data'], 
                                        batch_type=settings['batch_type'], batch_time=settings['batch_time'], 
                                        batch_time_frac=settings['batch_time_frac'],
                                        noise = 0,
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'])

    # Initialization
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                    log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
    odenet.float()
    param_count = sum(p.numel() for p in odenet.parameters() if p.requires_grad)
    param_ratio = round(param_count/ (data_handler.dim)**2, 3)
    print("Using a NN with {} neurons per layer, with {} trainable parameters, i.e. parametrization ratio = {}".format(settings['neurons_per_layer'], param_count, param_ratio))
    
    pretrained_model_file = '/home/ubuntu/phoenix/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
    odenet.load(pretrained_model_file)
        

    #RNAODE Rndom forest regression
    noise_to_check = 0
    dynamo_vf_inputs = get_true_val_velocities_new(odenet, data_handler, data_handler_velo, settings['method'], settings['batch_type'], 
                                                   noise_for_training = noise_to_check, scale_factor_for_counts = 1, breast = True)
    
    X_train = dynamo_vf_inputs['x_train']
    X_train_target = dynamo_vf_inputs['x_target_train']
    t_train = dynamo_vf_inputs['t_train']
    true_velos_train = dynamo_vf_inputs['true_velo_x_train']

    
    X_val = dynamo_vf_inputs['x_val']
    X_val_target = dynamo_vf_inputs['x_target_val']
    t_val = dynamo_vf_inputs['t_val']
    true_velos_val = dynamo_vf_inputs['true_velo_x_val']
    
    phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']
    
    X_full = dynamo_vf_inputs['x_full']
    true_velos_full = dynamo_vf_inputs['true_velo_x_full']

    X_test = dynamo_vf_inputs['x_test']
    X_test_target = dynamo_vf_inputs['x_target_test']
    t_test = dynamo_vf_inputs['t_test']


    print("..................................")
    print("PHX val corr vs true velos (w/o access!):", 
    round(np.corrcoef(true_velos_val.flatten(), phx_val_set_pred.flatten())[0,1], 4)) #this is just including to see.
    print("..................................")
    
    
    #quit()
    best_val_traj_mse = inf 
    best_n_trees = None
    best_rf = None
    
    for this_num_trees in [100]: #  , 200, 500, 750, 1000, 1200 1200, , 2000
        time_start = time.time()
        print("RNA ODE, num_trees:", this_num_trees)
        print("NOISE = {}".format(noise_to_check))
        rf_mod, train_score = BUILD_MODEL(counts = X_train, 
                                            velocity = true_velos_train,
                                            n_estimators=this_num_trees, 
                                            max_depth=None, 
                                            train_size=0.999)
        
        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))
        
    
        velo_fun_x = lambda t,x : rf_mod.predict(x.reshape(1, -1))
        
        pred_next_pts = pred_traj_given_ode(my_ode_func =velo_fun_x, 
                                                    X_val = X_val, 
                                                    t_val = t_val)

        mse_val_traj = np.mean((X_val_target- pred_next_pts)**2)
        print("MSE val traj = {:.5E}".format(mse_val_traj)) 

        
        if mse_val_traj < best_val_traj_mse:
            print("updating best Random Forest!")
            best_val_traj_mse = mse_val_traj
            best_trees = this_num_trees
            best_rf = rf_mod
            best_rf_func = velo_fun_x
        
        print("..................................")
        
    pred_next_pts = pred_traj_given_ode(my_ode_func = best_rf_func, 
                                            X_val = X_test, 
                                            t_val = t_test,
                                            breast_test = True)
        
    best_test_traj_mse = np.mean((X_test_target - pred_next_pts)**2)
    print("MSE test traj of best val model = {:.5E}".format(best_test_traj_mse))

    var_explained_pw = my_r_squared(pred_next_pts, X_test_target)
    print("R^2 test traj of best val model = {:.2%}".format(var_explained_pw))   
    print("Best nTrees:", best_trees, ", test traj MSE of that model:", best_test_traj_mse)

    N_list = [50, 100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 11165]
        
    loss_calcs = get_true_val_set_r2(pred_next_pts, X_test_target, "/home/ubuntu/phoenix/ode_net/code/model_inspect", N_list)

    f = open('{}/R2_by_Ngene_rnaode_{}.txt'.format("/home/ubuntu/phoenix/ode_net/code/model_inspect/", best_trees), 'w')
    print("__________________\n", file = f)
    for n, perf_couple in zip(N_list, loss_calcs):
        print("R^2, MSE, PercErr of val traj (init-val) for top {} genes: {:.2%}, {:.5E}, {:.2%}".format(n, perf_couple[0], perf_couple[1], perf_couple[2]), file = f)
    print("__________________\n", file = f)


    print("DONE! Now visualizing...")


    visualizer = Visualizator1D_new(data_handler, odenet, best_rf_func, settings, my_range_tuple = (0, 1.2))
    with torch.no_grad():
        visualizer.visualize()
        visualizer.plot()
        visualizer.save("/home/ubuntu/phoenix/ode_net/code/model_inspect/rnaode_BRCA_11165.png")
    
    print("obtaining GRN now..\n")
    my_GRN = GET_GRN(counts = X_full, velocity = true_velos_full, model_to_test = best_rf)
    np.savetxt("/home/ubuntu/phoenix/ode_net/code/model_inspect/effects_mat_rnaode.csv", my_GRN, delimiter=",")

