import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import torch
from torch import nn
import glob
import pandas as pd
import numpy as np
from functools import partial



def seed_torch(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def save_model(epochs, model, optimizer, loss_val, config, outfile):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'activation_fn': config['act'],
                'best_loss': loss_val,
                'batch_size': config['batch_size'],
                'h_dim': config['h_dim'],
                'num_rfchain': config['num_rfchain'],
                'num_meas': config['num_meas'],
                'conv_layer': config['conv_layer'],
                'mlp_layer': config['mlp_layer'],
                'dropout': config['dropout'],
                'num_states': config['num_states']
                }, outfile)
    

def save_testset(path, data_filename, test_data, dataset):
    graphs = []
    data_dict = {}
    pilots, combiner, channel, psn_dev = [],[],[],[]
    for n in range(len(test_data)):
        graphs.append(test_data[n][0])
        pilots.append(test_data[n][1])
        combiner.append(test_data[n][2])
        channel.append(test_data[n][3])
        psn_dev.append(test_data[n][4])
    
    data_dict['psn_dev'] = np.array(psn_dev)
    data_dict['channel'] = np.array(channel)
    data_dict['pilots'] = np.array(pilots)
    data_dict['combiner'] = np.array(combiner)
    data_dict['num_rfchain'] = dataset.Nrf
    data_dict['indices'] = dataset.indices
    data_dict['num_meas'] = dataset.Q
    data_dict['num_states'] = dataset.Nb
    data_dict['num_antenna'] = dataset.Nr
    data_dict['num_user'] = dataset.Nue
    # data_dict['pilot_len'] = dataset.pilot_len

    dgl.save_graphs(os.path.join(path, data_filename +'_test.dgl'), graphs)
    np.save(os.path.join(path, data_filename +'_test.npy'), data_dict)

    
    
class GraphDataset(DGLDataset):
    def __init__(self, filename, path="<full_path_name>/data/"): # NOTE: need to replace <...> with the full path name of the directory which contain the script
        super().__init__(name=filename, url=None, raw_dir=path)

    def process(self):
        self.graphs, _ = dgl.load_graphs(os.path.join(self.raw_dir, self.name +".dgl"))
        data_dict = np.load(os.path.join(self.raw_dir, self.name +".npy") ,allow_pickle=True).item()

        self.psn_dev = data_dict['psn_dev'] # PSN deviation matrices
        self.pilots = data_dict['pilots']
        self.channel = data_dict['channel']
        self.combiner = data_dict['combiner']
        self.Nrf = data_dict['num_rfchain']
        self.indices = data_dict['indices']
        self.Q = data_dict['num_meas']
        self.Nb = data_dict['num_states']
        self.Nr = data_dict['num_antenna']
        self.Nue = data_dict['num_user']
        
    def __getitem__(self, i):
        return self.graphs[i], self.pilots[i], self.combiner, self.channel[i], self.psn_dev[i]

    def __len__(self):
        return len(self.graphs)
    


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False