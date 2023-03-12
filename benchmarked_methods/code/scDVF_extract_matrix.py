# Imports
import sys
import os
import argparse
import inspect
from datetime import datetime
import numpy as np

import torch
import scipy
import tensorflow as tf
from tensorflow import keras

from scipy.integrate import odeint, solve_ivp

#from datagenerator import DataGenerator
from datahandler import DataHandler
from odenet import ODENet
from read_config import read_arguments_from_file
from visualization_inte import *
from helper_true_velo import *



def _build_save_file_name(save_path, epochs):
    return 'scDVF_{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_inte.cfg')
clean_name =  "chalmers_690genes_150samples_earlyT_0bimod_1initvar" 
clean_name_velo =  "chalmers_690genes_150samples_earlyT_0bimod_1initvar_DERIVATIVES" 
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
    for train_noise_level in [0 ,0.025, 0.05, 0.1]:
        dynamo_vf_inputs = get_true_val_velocities(odenet, data_handler, data_handler_velo, settings['method'], settings['batch_type'], noise_for_training= train_noise_level)
        
        X_train = dynamo_vf_inputs['x_train']
        X_val = dynamo_vf_inputs['x_val']
        X_val_target = dynamo_vf_inputs['x_target_val']
        X_full = dynamo_vf_inputs['x_full']
        Y_train = dynamo_vf_inputs['true_velo_x_train']
        Y_val = dynamo_vf_inputs['true_velo_x_val']
        Y_full = dynamo_vf_inputs['true_velo_x_full']
        t_val = dynamo_vf_inputs['t_val']
        phx_val_set_pred = dynamo_vf_inputs['phx_val_set_pred']
        

        print("..................................")
        print("PHX val corr vs true velos (w/o access!):", 
        round(np.corrcoef(Y_val.flatten(), phx_val_set_pred.flatten())[0,1], 4))
        #print("PHX val MSE vs true velos (w/o access!): {:.3E}".format(np.mean((Y_val.flatten() - phx_val_set_pred.flatten())**2)))
        
        
        print("creating scDVF VAE model..")
        input_dim = keras.Input(shape=(X_train.shape[1],))
        encoded = keras.layers.Dense(64, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(input_dim)
        encoded = keras.layers.Dense(64, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(encoded)
        encoded = keras.layers.Dense(64, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(encoded)
        encoded = keras.layers.Dense(16, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(encoded)

        decoded = keras.layers.Dense(16, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(encoded)
        decoded = keras.layers.Dense(64, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(decoded)
        decoded = keras.layers.Dense(64, activation="relu", activity_regularizer=keras.regularizers.l1(1e-6))(decoded)
        decoded = keras.layers.Dense(64, activity_regularizer=keras.regularizers.l1(1e-6))(decoded)
        decoded = keras.layers.Dense(Y_train.shape[1])(decoded)

        encoder = keras.Model(input_dim, encoded)
        autoencoder = keras.Model(input_dim, decoded)

        opt = keras.optimizers.Adam(learning_rate=0.00005)
        autoencoder.compile(optimizer=opt, loss='mse')
        #autoencoder.summary()
        print("done creating model, fitting now..")
        
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        autoencoder.fit(X_train, Y_train,
            epochs=100,
            batch_size=2,
            shuffle=True,
            validation_data=(X_val, Y_val),
            callbacks=[es], 
            verbose = 0)

        pred_val = autoencoder.predict(X_val, verbose = 0).flatten()    
        corr_coeff_val = round(np.corrcoef(Y_val.flatten(), 
                                            pred_val)[0,1], 4)
        pred_train = autoencoder.predict(X_train, verbose = 0).flatten()  
        corr_coeff_train = round(np.corrcoef(Y_train.flatten(), 
                                            pred_train)[0,1], 4)
        print("")
        print("train corr:", corr_coeff_train, ", val_corr:", corr_coeff_val)
        print("..................................")
        print("Now doing trajectory work...")

            
        velo_fun_x = lambda t,x :  np.squeeze(autoencoder.predict(np.expand_dims(x, axis = 0), 
                                                        verbose = 0))

        pred_next_pts = pred_traj_given_ode(my_ode_func = velo_fun_x, 
                                                    X_val = X_val, 
                                                    t_val = t_val)

        mse_val_traj = np.mean((X_val_target - pred_next_pts)**2)
        print("MSE val traj = {:.3E}".format(mse_val_traj))                                           
        print("..................................")
        
        # print("obtaining GRN now..\n")

        # def reverse_raw_ae(t, in_x):
        #     input_x = tf.convert_to_tensor(np.expand_dims(in_x, axis=0))
        #     dx = autoencoder.predict(input_x, verbose = 0).flatten() 
        #     return dx                                


        # def short_reverse_interpolate(y0, X_full, neigh, pca, umap_reducer, steps, intermediate_steps, noise_sd):
        #     solution = []
        #     for step in range(steps):
        #         # Interpolate using autoencoder
        #         t_eval = list(range(intermediate_steps))
        #         noise = np.random.normal(0, noise_sd)
        #         sol = solve_ivp(reverse_raw_ae, [min(t_eval), max(t_eval)], y0+noise, method="RK23", t_eval=t_eval)
        #         y = sol.y.T
                
        #         # Lower dimensionality
        #         ending_pt_pca = pca.transform(np.nan_to_num(np.log1p(y)))
                
        #         # Find knn reference points
        #         interp_neigh = neigh.kneighbors(ending_pt_pca)
                
        #         # New reference point
        #         y0 = np.median(X_full[interp_neigh[1][-1, :], :], axis=0)
        #         solution.append(y0)
            
        #     return np.array(solution)


        # # PCA the count data
        # pca = sklearn.decomposition.PCA(n_components=30)
        # adata_pca = pca.fit_transform(np.log1p(X_full))
        # # Further reduce the dim with UMAP
        # umap_reducer = umap.UMAP(random_state=42)
        # adata_umap = umap_reducer.fit_transform(adata_pca)
        # # Construct KNN with PCA
        # neigh = sklearn.neighbors.NearestNeighbors(n_neighbors=30)
        # neigh.fit(adata_pca)

        # n_cells = 200
        # num_steps = 15
        # cell_path = np.zeros((n_cells, num_steps, X_full.shape[1]))
        # for i in range(n_cells):
        #     print(i)
        #     y0 = np.random.rand(X_full.shape[1])
        #     y0_noise = 0
        #     # Solve for the cell with initial & velocity noise
        #     y_solution = short_reverse_interpolate(y0+y0_noise, X_full = X_full, 
        #                                                         neigh = neigh, pca = pca,
        #                                                         umap_reducer = umap_reducer, 
        #                                                         steps = num_steps,
        #                                                         intermediate_steps = 5, 
        #                                                         noise_sd = 0)
        #     cell_path[i] = y_solution
        
        
        # cell_path = pd.DataFrame(data=cell_path[:,2,:])
        # corr = cell_path.corr(method='pearson').fillna(0)
        # corr = corr.loc[:, (corr != 0).any(axis=0)]
        # corr = corr.loc[(corr != 0).any(axis=1), :]
        # np.savetxt("/home/ubuntu/neural_ODE/ode_net/code/model_inspect/effects_mat_{}.csv".format(noise_level), corr, delimiter=",")

        
    
    
