import torch
import torch.nn as nn
import sys
import numpy as np

from torch.nn.init import calculate_gain
#torch.set_num_threads(36)

def off_diag_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, LogSigProdLayer):
        with torch.no_grad():
            m.weight.copy_(torch.triu(m.weight, diagonal = 1) + torch.tril(m.weight, diagonal = -1))

def get_zero_grad_hook(mask):
    def hook(grad):
        return grad * mask
    return hook    


class SoftsignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class
        #self.shift = shift

    def forward(self, input):
        shift = 0.5
        shifted_input =(input- shift) #500*
        abs_shifted_input = torch.abs(shifted_input)
        return(shifted_input/(1+abs_shifted_input))   #1/500*

class LogShiftedSoftSignMod(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        shifted_input =  input - 0.5 
        abs_shifted_input = torch.abs(shifted_input)
        soft_sign_mod = shifted_input/(1+abs_shifted_input)
        return(torch.log1p(soft_sign_mod))  


class ODENet(nn.Module):
    ''' ODE-Net class implementation '''

    
    def __init__(self, device, ndim, explicit_time=False, neurons=100, log_scale = "linear", init_bias_y = 0):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()

        self.ndim = ndim
        self.explicit_time = explicit_time
        self.log_scale = log_scale
        self.init_bias_y = init_bias_y
        #only use first 68 (i.e. TFs) as NN inputs
        #in general should be num_tf = ndim
        self.num_tf = 73 
        
        # Create a new sequential model with ndim inputs and outputs
        if explicit_time:
            self.net = nn.Sequential(
                nn.Linear(ndim + 1, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, neurons),
                nn.LeakyReLU(),
                nn.Linear(neurons, ndim)
            )
        else: #6 layers
           
            
            self.net_ootb = nn.Sequential()
            self.net_ootb.add_module('activation_0', nn.Tanh())
            self.net_ootb.add_module('linear_1', nn.Linear(ndim, 2*neurons, bias = True))
            self.net_ootb.add_module('activation_1', nn.Tanh())
            self.net_ootb.add_module('linear_out', nn.Linear(2*neurons,ndim, bias = True))
        

                
        # Initialize the layers of the model
        for n in self.net_ootb.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05) 
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
        
       
                
        #self.net_prods.apply(off_diag_init)
        #self.net_sums.apply(off_diag_init)
        
      
        #creating masks and register the hooks
        #mask_prods = torch.tril(torch.ones_like(self.net_prods.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_prods.linear_out.weight), diagonal = 1)
        #mask_sums = torch.tril(torch.ones_like(self.net_sums.linear_out.weight), diagonal = -1) + torch.triu(torch.ones_like(self.net_sums.linear_out.weight), diagonal = 1)
        
        #self.net_prods.linear_out.weight.register_hook(get_zero_grad_hook(mask_prods))
        #self.net_sums.linear_out.weight.register_hook(get_zero_grad_hook(mask_sums)) 

        
        self.net_ootb.to(device)
        
        
    def forward(self, t, y):
        res = self.net_ootb(y)
        final = res - y
        return(final) 


    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        ootb_path = fp[:idx] + '_ootb' + fp[idx:]
        torch.save(self.net_ootb, ootb_path)
        

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def load_model(self, fp):
        ''' Load a model from a file '''
        idx = fp.index('.')
        ootb_path =  fp[:idx] + '_ootb' + fp[idx:]
        self.net_ootb = torch.load(ootb_path)
        self.net_ootb.to('cpu')
        
    def load(self, fp):
        ''' General loading from a file '''
        try:
            print('Trying to load model from file= {}'.format(fp))
            self.load_model(fp)
            print('Done')
        except:
            print('Failed! Trying to load parameters from file...')
            try:
                self.load_dict(fp)
                print('Done')
            except:
                print('Failed! Network structure is not correct, cannot load parameters from file, exiting!')
                sys.exit(0)

    def to(self, device):
        self.net.to(device)