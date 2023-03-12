import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as patches
from figure_saver import save_figure
import numpy as np
import torch
import random 

try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint


class Visualizator():

    def visualize(self):
        pass

    def __init__(self, data_handler, odenet, settings):
        self.data_handler = data_handler
        self.odenet = odenet
        self.settings = settings

    def save_plot(self, fig, folder, name):
        fig.savefig('{}/{}.eps'.format(folder, name))

class Visualizator1D(Visualizator):
    
    def __init__(self, data_handler, odenet, settings):
        super().__init__(data_handler, odenet, settings)
        # IH: uncomment this when vis dyn
        #self.fig_dyn = plt.figure(figsize=(6,6))
        #self.fig_dyn.canvas.set_window_title("Dynamics")
        #self.ax_dyn = self.fig_dyn.add_subplot(111, frameon=False)
        
        self.fig_traj_split = plt.figure(figsize=(15,15), tight_layout=True)
        self.fig_traj_split.canvas.set_window_title("Trajectories in each dimension")
        
        self.TOT_ROWS = 5
        self.TOT_COLS = 6

        if self.data_handler.batch_type == "single":
            self.sample_plot_val_cutoff = 0
        else:
            if self.data_handler.val_split !=1 :
                self.sample_plot_val_cutoff = min(self.data_handler.n_val, 2)
                #print( self.sample_plot_val_cutoff)
            else:
                self.sample_plot_val_cutoff = min(self.data_handler.n_val, 7)

        self.genes_to_viz = sorted(random.sample(range(self.data_handler.dim),30)) #only plot 30 genes
        
        #breast cancer genes 
        #self.genes_to_viz = [3106,7007, 556, 3072, 831, 1031, 1032, 5012, 6093] + sorted(random.sample(range(self.data_handler.dim),21)) #desmedt genes
        
        self.axes_traj_split = self.fig_traj_split.subplots(nrows=self.TOT_ROWS, ncols=self.TOT_COLS, sharex=False, sharey=True, subplot_kw={'frameon':True})
        
        self.legend_traj = [Line2D([0], [0], color='black', linestyle='-.', label='NN approx. of dynamics'),Line2D([0], [0], color='green', linestyle='-', label='True dynamics'),Line2D([0], [0], marker='o', color='red', label='Observed data', markerfacecolor='red', markersize=5)]
        #self.legend_traj = [Line2D([0], [0], color='blue', linestyle='-.', label='NN approx. of dynamics'),Line2D([0], [0], marker='o', color='grey', label='Observed data', markerfacecolor='red', markersize=5)]
        
        self.fig_traj_split.legend(handles=self.legend_traj, loc='upper center', ncol=3)

        self._set_ax_limits()

        #plt.show()
        #plt.savefig('initial_plot.png')
        
    def plot(self):
        #plt.figure(1)
        # IH: uncomment this when vis dyn
        #self.fig_dyn.canvas.draw_idle()
        #self.fig_dyn.canvas.start_event_loop(0.005)
        #plt.figure(2)
        self.fig_traj_split.canvas.draw_idle()
        self.fig_traj_split.canvas.start_event_loop(0.005)

    def _set_ax_limits(self):
        data = self.data_handler.data_np
        #times = self.extrap_timepoints
        times = self.data_handler.time_np
        self.EXTRA_WIDTH_TRAJ = 0.2
        self.EXTRA_WIDTH_DYN = 1

        #self.time_span = (np.min([np.min(time[:]) for time in times]),
        #                  np.max([np.max(time[:]) for time in times]))
        self.time_span = (0.0, 15)
        self.time_width = self.time_span[1] - self.time_span[0]

    


        log_scale = self.settings['log_scale']
        if log_scale == "log":
            upper_lim =  -0.2
            lower_lim = -5
        elif log_scale == "reciprocal":    
            upper_lim = 1.3
            lower_lim = 0.4
        else: #i.e. linear 
            upper_lim = 1.1 #6+14 
            lower_lim = -0.1 #-6+10 

        for row_num,this_row_plots in enumerate(self.axes_traj_split):
            for col_num, ax in enumerate(this_row_plots):
                ax.set_xlim((self.time_span[0]-self.time_width*self.EXTRA_WIDTH_TRAJ,
                            self.time_span[1]+self.time_width*self.EXTRA_WIDTH_TRAJ))
                ax.set_ylim((self.settings['scale_expression']*lower_lim,
                self.settings['scale_expression']*upper_lim))
         

    def visualize(self):
        self.trajectories, self.all_plotted_samples, self.extrap_timepoints = self.data_handler.calculate_trajectory(self.odenet, self.settings['method'], num_val_trajs = self.sample_plot_val_cutoff)
        self._visualize_trajectories_split()
        #self._visualize_dynamics()
        self._set_ax_limits()

    def _visualize_trajectories_split(self):
        times = self.data_handler.time_np
        data_np_to_plot = [self.data_handler.data_np[i] for i in self.all_plotted_samples]
        data_np_0noise_to_plot = [self.data_handler.data_np_0noise[i] for i in self.all_plotted_samples]

        for row_num,this_row_plots in enumerate(self.axes_traj_split):
            for col_num, ax in enumerate(this_row_plots):
                gene = self.genes_to_viz[row_num*self.TOT_COLS + col_num] #IH restricting to plot only few genes
                ax.cla()
                for sample_idx, (approx_traj, traj, true_mean) in enumerate(zip(self.trajectories, data_np_to_plot, data_np_0noise_to_plot)):
                    
                    if self.data_handler.n_val > 0 and sample_idx < self.sample_plot_val_cutoff:
                        plot_col = "red"
                    else:
                        plot_col = "blue"    
                    #ax.plot(times[sample_idx].flatten()[0:], approx_traj[:,:,gene].numpy().flatten(), color = plot_col, linestyle = "dashdot", lw=1) 
                    ax.plot( self.extrap_timepoints, approx_traj[:,:,gene].numpy().flatten(), color = plot_col, linestyle = "dashdot", lw=1) #
                    ax.plot(times[sample_idx].flatten(), traj[:,:,gene].flatten(), 'ko', alpha=0.2)
                    ax.plot(times[sample_idx].flatten(), true_mean[:,:,gene].flatten(),'g-', lw=1.5, alpha = 0.5) #
                   
                
                ax.set_xlabel(r'$t$')
        

    def save(self, dir, epoch):
         # IH: uncomment this when vis dyn
         #self.fig_dyn.savefig('{}dyn_epoch{}.png'.format(dir, epoch))
         self.fig_traj_split.savefig('{}viz_genes_epoch{}.png'.format(dir, epoch))