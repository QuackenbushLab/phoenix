# Imports
from cmath import inf
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np

import torch
import torch.optim as optim

import dynamo as dyn

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *
from helper_true_velo import *



def _build_save_file_name(save_path, epochs):
    return 'dynamo_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
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
    
    #DYNAMO vector field RKHS regression
    noise_level_to_check = 0
    dynamo_vf_inputs = get_true_val_velocities(odenet, data_handler, data_handler_velo,settings['method'], settings['batch_type'], noise_for_training = noise_level_to_check)
    
    X_train = dynamo_vf_inputs['x_train']
    X_val = dynamo_vf_inputs['x_val']
    X_val_target = dynamo_vf_inputs['x_target_val']
    t_val = dynamo_vf_inputs['t_val']
    true_velos_train = dynamo_vf_inputs['true_velo_x_train']
    true_velos_val = dynamo_vf_inputs['true_velo_x_val']
    phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']

    print("..................................")
    print("PHX val corr vs true velos (w/o access!):", 
    round(np.corrcoef(true_velos_val.flatten(), phx_val_set_pred.flatten())[0,1], 4))
    print("..................................")
    
    best_val_traj_mse = inf 
    best_M = None
    best_vf = None

    for this_M in [150, 200, 300, 400, 600]:

        print("Dynamo, M:", this_M)
        my_vf = dyn.vf.SvcVectorField(X = X_train, V = true_velos_train, 
                                        Grid = X_val,
                                        gamma = 1, M = this_M, lambda_ = 3) #gamma = 1 since we dont think there are any outliers
        trained_results = my_vf.train(normalize = False)
        dyn_pred_velos_train = trained_results['V']
        dyn_pred_velos_val = trained_results['grid_V']
        
        corr_coeff_train = round(np.corrcoef(true_velos_train.flatten(), 
                                        dyn_pred_velos_train.flatten())[0,1], 4)
        corr_coeff_val = round(np.corrcoef(true_velos_val.flatten(), 
                                        dyn_pred_velos_val.flatten())[0,1], 4)

                 
        
        
        print("train corr:", corr_coeff_train, ", val_corr:", corr_coeff_val)
        
        
       
        velo_fun_x = lambda t,x : my_vf.func(x)
        #velo_fun_x = lambda t,x : torch.squeeze(odenet.forward(torch.tensor([99,999]), torch.unsqueeze(torch.from_numpy(x), dim = 0).float())).detach().numpy()
       
        pred_next_pts = pred_traj_given_ode(my_ode_func = velo_fun_x, 
                                                    X_val = X_val, 
                                                    t_val = t_val)

        mse_val_traj = np.mean((X_val_target - pred_next_pts)**2)
        print("MSE val traj = {:.3E}".format(mse_val_traj)) 
        
        if mse_val_traj < best_val_traj_mse:
            print("updating best VF!")
            best_val_traj_mse = mse_val_traj
            best_M = this_M
            best_vf = my_vf

        #print(trained_results['C'].shape)
        print("..................................")

    
    print("Best M:", best_M,"best val traj MSE:", best_val_traj_mse)
    # print("obtaining Jacobian now..")

    # #Jacobian analysis

    # jac = best_vf.get_Jacobian()
    # n_iter = 10000
    # jac_sum = np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    # for iter in range(n_iter -1):
    #     jac_sum = jac_sum + np.abs(jac(np.random.uniform(low=0.0, high=1.0, size=data_handler.dim)))
    
    # jac_avg = np.transpose(jac_sum/n_iter)
    # np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat.csv", jac_avg, delimiter=",")

    # print("")
    # print("saved abosulte average jacobian matrix, taken over", n_iter, "random points.")
    # quit()
