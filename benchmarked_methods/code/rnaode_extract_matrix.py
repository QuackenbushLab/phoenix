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
from solve_eq import solve_eq
from visualization_inte import *

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
#from sklearn.tree import BaseDecisionTree

from GRN_rnaode import *
from helper_true_velo import *

def BUILD_MODEL(counts, velocity, genes=None, tfs=None, method='rf', n_estimators=10, max_depth=10, lasso_alpha=1, train_size=0.7):
    '''
    v = f(x, a). return fitted function f
    method: 'rf'(random forest) / 'lasso' / 'linear'.
    '''
    if genes is None:
        genes = np.array([True] * counts.shape[1])
    if tfs is None:
        tfs = genes
    x, x_val, y, y_val = train_test_split(counts[:, tfs], velocity[:, genes], 
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
    #test_score = (model.score(x_val, y_val))**0.5
    
    print('Fitted model | Training Corr: %.4f;' % (train_score))
    return model, train_score


def _build_save_file_name(save_path, epochs):
    return 'rnaode_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name =  "chalmers_350genes_150samples_earlyT_0bimod_1initvar" 
clean_name_velo =  "chalmers_350genes_150samples_earlyT_0bimod_1initvar_DERIVATIVES" 
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name))
parser.add_argument('--velo_data', type=str, default='/home/ubuntu/neural_ODE/ground_truth_simulator/clean_data/{}.csv'.format(clean_name_velo))


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

    # Create image and model save directory
    if not os.path.exists(output_root_dir):
        os.makedirs(output_root_dir, exist_ok=True)
    if not os.path.exists(img_save_dir):
        os.mkdir(img_save_dir)
    if not os.path.exists(interm_models_save_dir):
        os.mkdir(interm_models_save_dir)

    # Save the settings for future reference
    with open('{}/settings.csv'.format(output_root_dir), 'w') as f:
        f.write("Setting,Value\n")
        for key in settings.keys():
            f.write("{},{}\n".format(key,settings[key]))

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
    
    pretrained_model_file = '/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
    odenet.load(pretrained_model_file)
        
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
        net_file.write('\n\n\n')
        net_file.write(inspect.getsource(ODENet.forward))
        net_file.write('\n')

    #quit()

    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, odenet, settings)

    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    
    
    #RNAODE Rndom forest regression
    noise_to_check = 0.025
    dynamo_vf_inputs = get_true_val_velocities(odenet, data_handler, data_handler_velo, settings['method'], settings['batch_type'], noise_for_training= noise_to_check)
    
    X_train = dynamo_vf_inputs['x_train']
    X_val = dynamo_vf_inputs['x_val']
    X_val_target = dynamo_vf_inputs['x_target_val']
    t_val = dynamo_vf_inputs['t_val']
    true_velos_train = dynamo_vf_inputs['true_velo_x_train']
    true_velos_val = dynamo_vf_inputs['true_velo_x_val']
    phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']
    X_full = dynamo_vf_inputs['x_full']
    true_velos_full = dynamo_vf_inputs['true_velo_x_full']


    print("..................................")
    print("PHX val corr vs true velos (w/o access!):", 
    round(np.corrcoef(true_velos_val.flatten(), phx_val_set_pred.flatten())[0,1], 4))
    print("..................................")
    
    
    #quit()
    best_val_corr = 0
    best_n_trees = None
    best_rf = None
    
    for this_num_trees in [2000]: #100, 250, 500, 1000, , 1000, 2000
        time_start = time.time()
        print("RNA ODE, num_trees:", this_num_trees)
        rf_mod, train_score = BUILD_MODEL(counts = X_train, 
                                            velocity = true_velos_train,
                                            n_estimators=this_num_trees, 
                                            max_depth=None, 
                                            train_size=0.99)
        # if test_score > best_val_corr:
        #     print("updating best RF!")
        #     best_val_corr = test_score 
        #     best_n_trees = this_num_trees
        #     best_rf = rf_mod
        
        time_end = time.time()
        print("Elapsed time: %.2f seconds" % (time_end - time_start))
        print("..................................")

    
    # print("Best num trees:", best_n_trees ,"best val corr:", best_val_corr)
    
    velo_fun_x = lambda t,x : rf_mod.predict(x.reshape(1, -1))
       
    pred_next_pts = pred_traj_given_ode(my_ode_func = velo_fun_x, 
                                                    X_val = X_val, 
                                                    t_val = t_val)
    
    mse_val_traj = np.mean((X_val_target - pred_next_pts)**2)
    print("MSE val traj = {:.3E}".format(mse_val_traj)) 
    
    quit()
    print("obtaining GRN now..\n")
    my_GRN = GET_GRN(counts = X_full, velocity = true_velos_full, model_to_test = best_rf)
    np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", my_GRN, delimiter=",")

