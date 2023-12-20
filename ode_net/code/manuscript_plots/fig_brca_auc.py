import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = '/home/ubuntu/phoenix/all_manuscript_models/brca_compete.csv'
df = pd.read_csv(file_path)

# Extract data for each model
n_values = df['N']
dynamo_values = df['Dynamo']
rna_ode_values = df['RNA-ODE']
phoenix_values = df['PHOENIX']
scdvf_values = df['scDVF']
ootb_values = df['OOTB']
prior_values = df['Prior']

# Set markersize, border thickness, and colors
markersize = 12
border_thickness = 2
colors = {
    'PHOENIX': 'dodgerblue',
    'RNA-ODE': 'orange',
    'Dynamo': 'purple',
    'OOTB': 'saddlebrown',
    'scDVF': 'green',
    'Prior': 'black'
}

# Set custom x-axis tick labels
custom_x_ticks = [0, 2000, 4000, 6000, 8000, 10000, 11165]

# Plotting
plt.figure(figsize=(12, 8))

# Plot lines with markers
plt.plot(n_values, phoenix_values, marker='o', markersize=markersize, linestyle='-', color=colors['PHOENIX'], label='PHOENIX', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7)
plt.plot(n_values, rna_ode_values, marker='o', markersize=markersize, linestyle='-', color=colors['RNA-ODE'], label='RNA-ODE', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7)
#plt.plot(n_values, ootb_values, marker='o', markersize=markersize, linestyle='-', color=colors['OOTB'], label='OOTB', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7)
plt.plot(n_values, ootb_values, marker='o', markersize=markersize, linestyle='-', color=colors['OOTB'], label='Out-of-the-box NeuralODE', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7) 
plt.plot(n_values, dynamo_values, marker='o', markersize=markersize, linestyle='-', color=colors['Dynamo'], label='Dynamo', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7)
plt.plot(n_values, scdvf_values, marker='o', markersize=markersize, linestyle='-', color=colors['scDVF'], label='DeepVelo', linewidth=4, markeredgecolor='black', markeredgewidth=border_thickness, alpha=0.7)

# Plot the 'Prior' line
#plt.plot(n_values, prior_values, linestyle='--', color=colors['Prior'], label='Prior', linewidth=4, alpha=0.7)

# Add labels
plt.xlabel('$N$', fontsize=18)
# Make "Explainability AUC" text in ylabel bold using LaTeX formatting
plt.ylabel(r'$\mathbf{Explainability\ AUC}$ for subgraph of' + '\n'+ r'$N$-most variable genes', fontsize=18)

# Set custom x-axis tick labels
plt.xticks(custom_x_ticks, fontsize=15)

# Adjust axis tick label size and axis thickness
plt.yticks(fontsize=15)
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.gca().spines['top'].set_linewidth(2)
plt.gca().spines['right'].set_linewidth(2)

# Remove gridlines
plt.grid(False)

# Adjust legend text size
legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=18, frameon=False)

# Save the plot with the specified filename
output_directory = '/home/ubuntu/phoenix/ode_net/code/output/just_plots'
output_file_path = f'{output_directory}/brca_auc_by_N_plot_no_prior.png'
plt.savefig(output_file_path, bbox_inches='tight')

