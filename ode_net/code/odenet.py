import torch
import torch.nn as nn
import sys
import numpy as np

from torch.nn.init import calculate_gain
#torch.set_num_threads(36)


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

    
    def __init__(self, device, ndim, explicit_time=False, neurons=100):
        ''' Initialize a new ODE-Net '''
        super(ODENet, self).__init__()

        self.ndim = ndim
        self.explicit_time = explicit_time
        
        self.net_prods = nn.Sequential()
        self.net_prods.add_module('activation_0',  LogShiftedSoftSignMod()) #
        self.net_prods.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))
        
        self.net_sums = nn.Sequential()
        self.net_sums.add_module('activation_0', SoftsignMod())
        self.net_sums.add_module('linear_out', nn.Linear(ndim, neurons, bias = True))

        self.net_alpha_combine = nn.Sequential()
        self.net_alpha_combine.add_module('linear_out',nn.Linear(2*neurons, ndim, bias = False))
        
        
        self.gene_multipliers = nn.Parameter(torch.rand(1,ndim), requires_grad= True)
                
        # Initialize the layers of the model
        for n in self.net_sums.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05) 
        
        for n in self.net_prods.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)
                
                
        for n in self.net_alpha_combine.modules():
            if isinstance(n, nn.Linear):
                nn.init.sparse_(n.weight,  sparsity=0.95, std = 0.05)
                #nn.init.orthogonal_(n.weight, gain = calculate_gain("sigmoid"))
    
        
        self.net_prods.to(device)
        self.gene_multipliers.to(device)
        self.net_sums.to(device)
        self.net_alpha_combine.to(device)
        
        
    def forward(self, t, y):
        sums = self.net_sums(y)
        prods = torch.exp(self.net_prods(y))
        sums_prods_concat = torch.cat((sums, prods), dim= - 1)
        joint = self.net_alpha_combine(sums_prods_concat)
        final = torch.relu(self.gene_multipliers)*(joint-y)
        return(final) 

    def prior_only_forward(self, t, y):
        sums = self.net_sums(y)
        prods = torch.exp(self.net_prods(y))
        sums_prods_concat = torch.cat((sums, prods), dim= - 1)
        joint = self.net_alpha_combine(sums_prods_concat)
        return(joint)

    def save(self, fp):
        ''' Save the model to file '''
        idx = fp.index('.')
        alpha_comb_path = fp[:idx] + '_alpha_comb' + fp[idx:]
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        torch.save(self.net_prods, prod_path)
        torch.save(self.net_sums, sum_path)
        torch.save(self.net_alpha_combine, alpha_comb_path)
        torch.save(self.gene_multipliers, gene_mult_path)
        

    def load_dict(self, fp):
        ''' Load a model from a dict file '''
        self.net.load_state_dict(torch.load(fp))
    
    def load_model(self, fp):
        ''' Load a model from a file '''
        idx = fp.index('.pt')
        gene_mult_path = fp[:idx] + '_gene_multipliers' + fp[idx:]
        prod_path =  fp[:idx] + '_prods' + fp[idx:]
        sum_path = fp[:idx] + '_sums' + fp[idx:]
        alpha_comb_path = fp[:idx] + '_alpha_comb' + fp[idx:]
        self.net_prods = torch.load(prod_path)
        self.net_sums = torch.load(sum_path)
        self.gene_multipliers = torch.load(gene_mult_path)
        self.net_alpha_combine = torch.load(alpha_comb_path)
        
        self.net_prods.to('cpu')
        self.net_sums.to('cpu')
        self.gene_multipliers.to('cpu')
        self.net_alpha_combine.to('cpu')

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