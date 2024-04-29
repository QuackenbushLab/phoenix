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

def soft_sign_mod(this_x):
    shift = 0.5
    shifted_input =(this_x- shift) #500*
    abs_shifted_input = torch.abs(shifted_input)
    return(shifted_input/(1+abs_shifted_input))   #

def plot_MSE(epoch_so_far, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir):
    
    # Create two subplots, one for the main MSE loss plot and one for the prior loss plot.
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    fig.set_size_inches(12, 6)

    ax1.plot(range(1, epoch_so_far + 1), training_loss, color="blue", label="Training loss")
    ax1.plot(range(1, epoch_so_far + 1), true_mean_losses, color="green", label="Noiseless test loss")
    
    if len(validation_loss) > 0:
        ax1.plot(range(1, epoch_so_far + 1), validation_loss, color="red", label="Validation loss")

    ax2.plot(range(1, epoch_so_far + 1), prior_losses, color="magenta", label="Prior loss")

    ax1.set_yscale('log')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Error (MSE)")
    ax1.legend(loc='upper right')

    ax2.set_yscale('log')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error (MSE)")
    ax2.set_title("Prior Loss")

    #plt.subplots_adjust(wspace=0.3)
    fig.tight_layout()
    plt.savefig("{}/MSE_loss.png".format(img_save_dir))
    np.savetxt('{}full_loss_info.csv'.format(output_root_dir), np.c_[training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based], delimiter=',')

def my_r_squared(output, target):
    x = output
    y = target
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    my_corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return(my_corr**2)

def get_true_val_set_r2(odenet, data_handler, method, batch_type):
    data_pw, t_pw, target_pw = data_handler.get_true_mu_set_pairwise(val_only = True, batch_type =  "single")
    with torch.no_grad():
        predictions_pw = torch.zeros(data_pw.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t_pw, data_pw)):
            predictions_pw[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] 
        var_explained_pw = my_r_squared(predictions_pw, target_pw)
        true_val_mse = torch.mean((predictions_pw - target_pw)**2)
        
    #data, t, target = data_handler.get_true_mu_set_init_val_based(val_only = True) 
        #predictions = torch.zeros(target.shape).to(data_handler.device)
        #for index, (time, batch_point) in enumerate(zip(t, data)):
        #    predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1:] 
        #var_explained_init_val_based = my_r_squared(predictions, target)
    
    return [var_explained_pw, true_val_mse]




def read_prior_matrix(prior_mat_file_loc, sparse = False, num_genes = 11165):
    if sparse == False: 
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        mat_torch = torch.from_numpy(mat)
        return mat_torch.float()
    else: #when scaling up >10000
        mat = np.genfromtxt(prior_mat_file_loc,delimiter=',')
        sparse_mat = torch.sparse_coo_tensor([mat[:,0].astype(int)-1, mat[:,1].astype(int)-1], mat[:,2], ( num_genes,  num_genes))
        mat_torch = sparse_mat.to_dense().float()
        return(mat_torch)



def validation(odenet, data_handler, method, explicit_time):
    data, t, target_full, n_val = data_handler.get_validation_set()
    if method == "trajectory":
        False

    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = []
        targets = []
        # For now we have to loop through manually, their implementation of odenet can only take fixed time lists.
        for index, (time, batch_point, target_point) in enumerate(zip(t, data, target_full)):
            #IH: 9/10/2021 - added these to handle unequal time availability 
            #comment these out when not requiring nan-value checking
            #not_nan_idx = [i for i in range(len(time)) if not torch.isnan(time[i])]
            #time = time[not_nan_idx]
            #not_nan_idx.pop()
            #batch_point = batch_point[not_nan_idx]
            #target_point = target_point[not_nan_idx]
            
            # Do prediction
            predictions.append(odeint(odenet, batch_point, time, method=method)[1])
            targets.append(target_point) #IH comment
            #predictions[index, :, :] = odeint(odenet, batch_point[0], time, method=method)[1:]

        # Calculate validation loss
        predictions = torch.cat(predictions, dim = 0).to(data_handler.device) #IH addition
        targets = torch.cat(targets, dim = 0).to(data_handler.device) 
        #loss = torch.mean((predictions - targets) ** 2) #regulated_loss(predictions, target, t, val = True)
        loss = torch.mean((predictions - targets)**2)
        #print("gene_mult_mean =", torch.mean(torch.relu(odenet.gene_multipliers) + 0.1))        
    return [loss, n_val]

def true_loss(odenet, data_handler, method):
    return [0,0]
    data, t, target = data_handler.get_true_mu_set() #tru_mu_prop = 1 (incorporate later)
    init_bias_y = data_handler.init_bias_y
    #odenet.eval()
    with torch.no_grad():
        predictions = torch.zeros(data.shape).to(data_handler.device)
        for index, (time, batch_point) in enumerate(zip(t, data)):
            predictions[index, :, :] = odeint(odenet, batch_point, time, method=method)[1] + init_bias_y #IH comment
        
        # Calculate true mean loss
        loss =  [torch.mean(torch.abs((predictions - target)/target)),torch.mean((predictions - target) ** 2)] #regulated_loss(predictions, target, t)
    return loss


def decrease_lr(opt, verbose, tot_epochs, epoch, lower_lr,  dec_lr_factor ):
    dir_string = "Decreasing"
    for param_group in opt.param_groups:
        param_group['lr'] = param_group['lr'] * dec_lr_factor
    if verbose:
        print(dir_string,"learning rate to: %f" % opt.param_groups[0]['lr'])


def training_step(odenet, data_handler, opt, method, batch_size, explicit_time, relative_error, batch_for_prior, prior_grad, loss_lambda):
    #print("Using {} threads training_step".format(torch.get_num_threads()))
    batch, t, target = data_handler.get_batch(batch_size)
    
    '''
    not_nan_idx = [i for i in range(len(t)) if not torch.any(torch.isnan(t[i]))]
    t = t[not_nan_idx]
    batch = batch[not_nan_idx]
    target = target[not_nan_idx]
    '''

    init_bias_y = data_handler.init_bias_y
    opt.zero_grad()
    predictions = torch.zeros(batch.shape).to(data_handler.device)
    for index, (time, batch_point) in enumerate(zip(t, batch)):
        predictions[index, :, :] = odeint(odenet, batch_point, time, method= method  )[1] + init_bias_y #IH comment
    
    loss_data = torch.mean((predictions - target)**2) 
    
    pred_grad = odenet.prior_only_forward(t,batch_for_prior)
    loss_prior = torch.mean((pred_grad - prior_grad)**2)
    #loss_prior = loss_data

    composed_loss = loss_lambda * loss_data + (1- loss_lambda) * loss_prior
    composed_loss.backward() #MOST EXPENSIVE STEP!
    opt.step()
    return [loss_data, loss_prior]

def _build_save_file_name(save_path, epochs):
    return '{}-{}-{}({};{})_{}_{}epochs'.format(str(datetime.now().year), str(datetime.now().month),
        str(datetime.now().day), str(datetime.now().hour), str(datetime.now().minute), save_path, epochs)

def save_model(odenet, folder, filename):
    odenet.save('{}{}.pt'.format(folder, filename))

parser = argparse.ArgumentParser('Testing')
parser.add_argument('--settings', type=str, default='config_breast.cfg')
clean_name =  "desmedt_500genes_1sample_178T" 
parser.add_argument('--data', type=str, default='/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/{}.csv'.format(clean_name))
test_data_name = "desmedt_500genes_1TESTsample_8middleT" 
parser.add_argument('--test_data', type=str, default='/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/{}.csv'.format(test_data_name))

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
                                        noise = settings['noise'],
                                        img_save_dir = img_save_dir,
                                        scale_expression = settings['scale_expression'],
                                        log_scale = settings['log_scale'],
                                        init_bias_y = settings['init_bias_y'],
                                        fp_test = args.test_data)
    
    abs_prior = True
    
    #Read in the prior matrix
    prior_mat_loc = '/home/ubuntu/neural_ODE/breast_cancer_data/clean_data/edge_prior_matrix_desmedt_500.csv'
    prior_mat = read_prior_matrix(prior_mat_loc, sparse = False, num_genes = data_handler.dim)
    
    if abs_prior:
        prior_mat = torch.abs(prior_mat)

    K = 10000
    batch_for_prior = (torch.rand(K,1,prior_mat.shape[0], device = data_handler.device)- 0.5)*1
    prior_grad = torch.matmul(batch_for_prior,prior_mat) #can be any model here that predicts the derivative

    del prior_mat

    #curriculum learning
    loss_lambda_at_start =  1
    loss_lambda_at_middle = 1
    loss_lambda_at_end = 1

    curriculum_epochs_1 = 7
    curriculum_epochs_2 = 20
    
    
    
    # Initialization
    odenet = ODENet(device, data_handler.dim, explicit_time=settings['explicit_time'], neurons = settings['neurons_per_layer'], 
                    log_scale = settings['log_scale'], init_bias_y = settings['init_bias_y'])
    odenet.float()
    param_count = sum(p.numel() for p in odenet.parameters() if p.requires_grad)
    param_ratio = round(param_count/ (data_handler.dim)**2, 3)
    print("Using a NN with {} neurons per layer, with {} trainable parameters, i.e. parametrization ratio = {}".format(settings['neurons_per_layer'], param_count, param_ratio))
    
    if settings['pretrained_model']:
        pretrained_model_file = '/home/ubuntu/neural_ODE/ode_net/code/output/_pretrained_best_model/best_val_model.pt'
        odenet.load(pretrained_model_file)
        #print("Loaded in pre-trained model!")
        
    with open('{}/network.txt'.format(output_root_dir), 'w') as net_file:
        net_file.write(odenet.__str__())
        net_file.write('\n\n\n')
        net_file.write(inspect.getsource(ODENet.forward))
        net_file.write('\n')
        if abs_prior:
            net_file.write('prior_mat = torch.abs(prior_mat)')
            net_file.write('\n')
        net_file.write('lambda at start (first {} epochs) = {}'.format(curriculum_epochs_1, loss_lambda_at_start))
        net_file.write('\n')
        net_file.write('lambda in middle (upto {} epochs) = {}'.format(curriculum_epochs_2, loss_lambda_at_middle))
        net_file.write('\n')
        net_file.write('and then lambda = {}'.format(loss_lambda_at_end))
        net_file.write('\n')
        #net_file.write('SPECIAL SOFTSIGN-transformed inputs to prior!')
        
        

    #quit()

    # Select optimizer
    print('Using optimizer: {}'.format(settings['optimizer']))
    if settings['optimizer'] == 'rmsprop':
        opt = optim.RMSprop(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'sgd':
        opt = optim.SGD(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    elif settings['optimizer'] == 'adagrad':
        opt = optim.Adagrad(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
    else:
#       opt = optim.Adam(odenet.parameters(), lr=settings['init_lr'], weight_decay=settings['weight_decay'])
        num_gene = data_handler.dim
        opt = optim.Adam([
                {'params': odenet.net_sums.linear_out.weight}, 
                {'params': odenet.net_sums.linear_out.bias},
                {'params': odenet.net_prods.linear_out.weight},
                {'params': odenet.net_prods.linear_out.bias},
                {'params': odenet.net_alpha_combine.linear_out.weight},
                {'params': odenet.gene_multipliers,'lr': 5*settings['init_lr']}
                
            ],  lr=settings['init_lr'], weight_decay=settings['weight_decay'])


    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', 
    factor=0.9, patience=6, threshold=1e-07, 
    threshold_mode='abs', cooldown=0, min_lr=0, eps=1e-09, verbose=True)

    
    # Init plot
    if settings['viz']:
        visualizer = Visualizator1D(data_handler, odenet, settings)

    # Training loop
    #batch_times = [] 
    epoch_times = []
    total_time = 0
    validation_loss = []
    training_loss = []
    prior_losses = []
    true_mean_losses = []
    true_mean_losses_init_val_based = []
    A_list = []

    min_loss = 0
    if settings['batch_type'] == 'single':
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])
    elif settings['batch_type'] == 'trajectory':
        iterations_in_epoch = data_handler.train_data_length
    else:
        iterations_in_epoch = ceil(data_handler.train_data_length / settings['batch_size'])

    if settings['viz']:
        with torch.no_grad():
            visualizer.visualize()
            visualizer.plot()
            visualizer.save(img_save_dir, 0)
    start_time = perf_counter()
    #quit()
    
    tot_epochs = settings['epochs']
    #viz_epochs = [round(tot_epochs*1/5), round(tot_epochs*2/5), round(tot_epochs*3/5), round(tot_epochs*4/5),tot_epochs]
    rep_epochs = [1, 5, 7, 10, 15, 20, 30, 40, 50, 75, 90, 100, 120, 150, 180, 200, tot_epochs]
    viz_epochs = rep_epochs
    zeroth_drop_done = False
    first_drop_done = False 
    second_drop_done = False
    rep_epochs_train_losses = []
    rep_epochs_val_losses = []
    rep_epochs_mu_losses = []
    rep_epochs_time_so_far = []
    rep_epochs_so_far = []
    consec_epochs_failed = 0
    epochs_to_fail_to_terminate = 40#15
    all_lrs_used = []

    #print(get_true_val_set_r2(odenet, data_handler, settings['method'], settings['batch_type']))
    
    for epoch in range(1, tot_epochs + 1):
            
        start_epoch_time = perf_counter()
        iteration_counter = 1
        data_handler.reset_epoch()
        #visualizer.save(img_save_dir, epoch) #IH added to test
        this_epoch_total_train_loss = 0
        this_epoch_total_prior_loss = 0
        print()
        print("[Running epoch {}/{}]".format(epoch, settings['epochs']))

        
        if epoch < curriculum_epochs_1:
            loss_lambda = loss_lambda_at_start 
        elif epoch >= curriculum_epochs_1 and epoch < curriculum_epochs_2:
            loss_lambda = loss_lambda_at_middle
        else:
            loss_lambda = loss_lambda_at_end    
        print("current loss_lambda =", loss_lambda)
        
        
        if settings['verbose']:
            pbar = tqdm(total=iterations_in_epoch, desc="Training loss:")
        while not data_handler.epoch_done:
            start_batch_time = perf_counter()
            
            loss_list = training_step(odenet, data_handler, opt, settings['method'], settings['batch_size'], settings['explicit_time'], settings['relative_error'], batch_for_prior, prior_grad, loss_lambda)
            loss = loss_list[0]
            prior_loss = loss_list[1]
            #batch_times.append(perf_counter() - start_batch_time)

            this_epoch_total_train_loss += loss.item()
            this_epoch_total_prior_loss += prior_loss.item()
            # Print and update plots
            iteration_counter += 1

            if settings['verbose']:
                pbar.update(1)
                pbar.set_description("Training loss, Prior loss: {:.2E}, {:.2E}".format(loss.item(), prior_loss.item()))
        
        epoch_times.append(perf_counter() - start_epoch_time)

        #Epoch done, now handle training loss
        train_loss = this_epoch_total_train_loss/iterations_in_epoch
        training_loss.append(train_loss)
        prior_losses.append(this_epoch_total_prior_loss/iterations_in_epoch)
        #print("Overall training loss {:.5E}".format(train_loss))

        mu_loss = get_true_val_set_r2(odenet, data_handler, settings['method'], settings['batch_type'])
        #mu_loss = true_loss(odenet, data_handler, settings['method'])
        true_mean_losses.append(mu_loss[1])
        true_mean_losses_init_val_based.append(mu_loss[0])
        all_lrs_used.append(opt.param_groups[0]['lr'])
        
        if epoch == 1:
                min_train_loss = train_loss
        else:
            if train_loss < min_train_loss:
                min_train_loss = train_loss
                true_loss_of_min_train_model =  mu_loss[1]
                #save_model(odenet, output_root_dir, 'best_train_model')
        


        if settings['verbose']:
            pbar.close()

        if settings['solve_A']:
            A = solve_eq(odenet, settings['solve_eq_gridsize'], (-5, 5, 0, 10, -3, 3, -10, 10))
            A_list.append(A)
            print('A =\n{}'.format(A))

        #handle true-mu loss
       
        if data_handler.n_val > 0:
            val_loss_list = validation(odenet, data_handler, settings['method'], settings['explicit_time'])
            val_loss = val_loss_list[0]
            validation_loss.append(val_loss)
            if epoch == 1:
                min_val_loss = val_loss
                true_loss_of_min_val_model = mu_loss[1]
                print('Model improved, saving current model')
                best_vaL_model_so_far = odenet
                save_model(odenet, output_root_dir, 'best_val_model')
            else:
                if val_loss < min_val_loss:
                    consec_epochs_failed = 0
                    min_val_loss = val_loss
                    true_loss_of_min_val_model =  mu_loss[1]
                    #saving true-mean loss of best val model
                    print('Model improved, saving current model')
                    save_model(odenet, output_root_dir, 'best_val_model')
                else:
                    consec_epochs_failed = consec_epochs_failed + 1

                    
            print("Validation loss {:.5E}, using {} points".format(val_loss, val_loss_list[1]))
            scheduler.step(val_loss)

        print("Overall training loss {:.5E}".format(train_loss))

        print("True MSE of val traj (pairwise): {:.5E}".format(mu_loss[1]))
        print("True R^2 of val traj (pairwise): {:.2%}".format(mu_loss[0]))

            
        if (settings['viz'] and epoch in viz_epochs) or (settings['viz'] and epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print("Saving plot")
            with torch.no_grad():
                #print("nope..")
                visualizer.visualize()
                visualizer.plot()
                visualizer.save(img_save_dir, epoch)
        
        #print("Saving intermediate model")
        #save_model(odenet, intermediate_models_dir, 'model_at_epoch{}'.format(epoch))
    
        # Decrease learning rate if specified
        if settings['dec_lr'] : #and epoch % settings['dec_lr'] == 0
            decrease_lr(opt, True,tot_epochs= tot_epochs,
             epoch = epoch, lower_lr = settings['init_lr'], dec_lr_factor = settings['dec_lr_factor'])
        
        '''
        #Decrease learning rate as a one-time thing:
        if (train_loss < 9*10**(-3) and zeroth_drop_done == False) or (epoch == 25 and zeroth_drop_done == False):
            decrease_lr(opt, settings['verbose'], one_time_drop= 5*10**(-3))
            zeroth_drop_done = True

        if (train_loss < 9*10**(-4) and first_drop_done == False) or (epoch == 50 and first_drop_done == False):
            decrease_lr(opt, settings['verbose'], one_time_drop= 1*10**(-3))
            first_drop_done = True
        
        if (train_loss < 2*10**(-4) and second_drop_done == False)  or (epoch == 75 and second_drop_done == False):
            decrease_lr(opt, settings['verbose'], one_time_drop= 1*10**(-4))
            second_drop_done = True
        '''
            
        #val_loss < (0.01 * settings['scale_expression'])**1
        if (epoch in rep_epochs) or (consec_epochs_failed == epochs_to_fail_to_terminate):
            print()
            rep_epochs_so_far.append(epoch)
            print("Epoch=", epoch)
            rep_time_so_far = (perf_counter() - start_time)/3600
            print("Time so far= ", rep_time_so_far, "hrs")
            rep_epochs_time_so_far.append(rep_time_so_far)
            print("Best training (MSE) so far= ", min_train_loss)
            rep_epochs_train_losses.append(min_train_loss)
            if data_handler.n_val > 0:
                print("Best validation (MSE) so far = ", min_val_loss.item())
                #print("True loss of best validation model (MSE) = ", true_loss_of_min_val_model.item())
                rep_epochs_val_losses.append(min_val_loss.item())
                #rep_epochs_mu_losses.append(0)
                rep_epochs_mu_losses.append(true_loss_of_min_val_model.item())
            else:
                #print("True loss of best training model (MSE) = ", true_loss_of_min_train_model.item())
                print("True loss of best training model (MSE) = ", 0)
            print("Saving MSE plot...")
            plot_MSE(epoch, training_loss, validation_loss, true_mean_losses, true_mean_losses_init_val_based, prior_losses, img_save_dir)    
            
            if settings['lr_range_test']:
                plot_LR_range_test(all_lrs_used, training_loss, img_save_dir)

            print("Saving losses..")
            if data_handler.n_val > 0:
                L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_val_losses, rep_epochs_mu_losses]
                np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
            
            print("Saving best intermediate val model..")
            interm_model_file_name = 'trained_model_epoch_' + str(epoch)
            save_model(odenet, interm_models_save_dir , interm_model_file_name)
                
            
            #else:
            #    L = [rep_epochs_so_far, rep_epochs_time_so_far, rep_epochs_train_losses, rep_epochs_mu_losses]
            #    np.savetxt('{}rep_epoch_losses.csv'.format(output_root_dir), np.transpose(L), delimiter=',')    
           
        

        if consec_epochs_failed==epochs_to_fail_to_terminate:
            print("Went {} epochs without improvement; terminating.".format(epochs_to_fail_to_terminate))
            break

        #if data_handler.n_val > 0 & val_loss < (0.01 * settings['scale_expression'])**1:
        #    print("SUCCESS! Reached validation target; terminating.")
        #    break    

    total_time = perf_counter() - start_time

    
    save_model(odenet, output_root_dir, 'final_model')

    print("Saving times")
    np.savetxt('{}epoch_times.csv'.format(output_root_dir), epoch_times, delimiter=',')

        
    print("DONE!")

  
 