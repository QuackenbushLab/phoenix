import torch
import numpy as np
import csv
try:
    from torchdiffeq.__init__ import odeint_adjoint as odeint
except ImportError:
    from torchdiffeq import odeint_adjoint as odeint

def expression_maker(val):
    #return(1/6 *(val + 3))
    return(val)

def readcsv(fp, device, noise_to_add, scale_expression):
    print("Reading from file {}".format(fp))
    print("Adding requested noise of {}".format(noise_to_add))
    print("Scaling gene-expression values by {} fold".format(scale_expression))
    data_np = []
    data_pt = []
    data_np_0noise = []
    data_pt_0noise = []
    t_np = []
    t_pt = []
    with open(fp, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        rows = []
        for r in reader:
            with_NA_strings = [float(string) if string != "" else float("NaN") for string in r]
            rows.append(with_NA_strings)
        dim = int(rows[0][0])
        ntraj = int(rows[0][1])
        data = rows[1:]
        #print(ntraj)
        for traj in range(ntraj):
            current_length = len(data[traj*(dim+1)])
            traj_data = np.zeros((current_length, 1, dim), dtype=np.float32)
            traj_data_0noise = np.zeros((current_length, 1, dim), dtype=np.float32)
            for d in range(dim + 1):
                if d == dim:
                    # Row is time data
                    row = [float(f) for f in data[traj*(dim+1) + d]]
                    t_np.append(np.array(row))
                    t_pt.append(torch.tensor(row).to(device))
                else:
                    #row is gene-expression data; so add noise here!
                    row = [expression_maker(float(f)+ np.random.normal(0, noise_to_add)) for f in data[traj*(dim+1) + d]]
                    traj_data[:,:,d] = scale_expression*np.expand_dims(np.array(row), axis=1)
                    row_0noise =  [expression_maker(float(f)) for f in data[traj*(dim+1) + d]]
                    traj_data_0noise[:,:,d] = scale_expression*np.expand_dims(np.array(row_0noise), axis=1)
            
            data_np.append(traj_data)
            data_np_0noise.append(traj_data_0noise)
            data_pt.append(torch.tensor(traj_data).to(device))
            data_pt_0noise.append(torch.tensor(traj_data_0noise).to(device))
           

    return data_np, data_pt, t_np, t_pt, dim, ntraj, data_np_0noise,data_pt_0noise

def writecsv(fp, dim, ntraj, data_np, t_np):
    ''' Write data from a datagenerator to a file '''
    # Clear the file
    f = open(fp, "w+")
    f.close()

    # Write the data into the file
    info = np.array([dim, ntraj])
    with open(fp, 'a') as f:
        writer = csv.writer(f, delimiter=',', lineterminator='\n')
        writer.writerow(info)
        for i in range(ntraj):
            for j in range(dim):
                writer.writerow(data_np[i][:,:,j].flatten())
            writer.writerow(t_np[i])
    print("Written to file {}".format(fp))
