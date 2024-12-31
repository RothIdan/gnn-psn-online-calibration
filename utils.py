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



class LossFn:
    """
    Loss function class.
    Calculate the error between the estimated deviation and the ground truth.
        We allow for error calculation when considering two phase ambiguities: 
            1. None, error in absolute - omega_hat_i = omega_i + err.
            2. global phase offset (delay) beta - omega_hat_i = omega_i + beta + err.
            3. global phase offset (delay) beta and phase slope alpha  - omega_hat_i = omega_i + alpha*i + beta + err (additive affine curve).
        Parameters:
            error_mode (str) - 'none' for 1, 'offset' for 2, 'affine' for 3.
            size (int) - number of phase shifters.
            p_states (int) - number of phase states at each phase shifter Nb
    """

    def __init__(self, error_mode, size, device, p_states=None):
        if error_mode not in ["absolute","offset","affine"]:
            raise KeyError(f'Error type {error_mode} not supported.\n'
                            'Use "absolute", "offset", or "affine" only') 
        
        self.mode = error_mode
        self.size = size
        self.modeified_pred = None
        self.device = device
        self.p_states = p_states

        if p_states is None: # "Matrix" form
            if error_mode == 'affine':
                X = torch.concatenate((torch.ones((size,1)),torch.arange(size).reshape(size,1)), dim=1)
                self.lhs_mat = (torch.inverse(X.T @ X) @ X.T).to(device)
            elif error_mode == 'offset':
                X = torch.ones((size,1))
                self.lhs_mat = (torch.inverse(X.T @ X) @ X.T).to(device)
        
        else: # "Tensor" form
            if error_mode == 'affine':
                X = torch.concatenate((torch.ones((size,1)),torch.arange(size).reshape(size,1)), dim=1)
                X = X.repeat((p_states,1))
                self.lhs_mat = (torch.inverse(X.T @ X) @ X.T).to(device)
            elif error_mode == 'offset':
                X = torch.ones((size,1))
                X = X.repeat((p_states,1))
                self.lhs_mat = (torch.inverse(X.T @ X) @ X.T).to(device)

        
        self.loss_fn = self._get_loss_fn(error_mode)
    
    
    def loss(self, pred, psn_dev):
        y = (pred - psn_dev).T # NOTE: make sure the order is 1st phase state for all antenna elements, 2nd phase states for all antenna elements,...
                               # pred = [w_1^1, w_2^1,..., w_Nr^1, w_1^2,..., w_Nr^2,..., w_1^Nb,..., w_Nr^Nb]^T

        return self.loss_fn(y, pred, psn_dev)
    
    
    def get_modified_pred(self):
        return self.modeified_pred


    def _get_loss_fn(self, mode):
        if mode == 'affine':
            return self._affine_loss
        elif mode == 'offset':
            return self._offset_loss
        else: # mode == 'absolute'
            return self._absolute_loss
            

    def _affine_loss(self, y, pred, psn_dev):
        p = self.lhs_mat @ y
        beta_hat, alpha_hat = p.T[:,0:1], p.T[:,1:2]
        # Eliminate ambiguity
        r = torch.arange(self.size, device=self.device)
        if self.p_states is not None: # "Tensor" form
            r = r.repeat(self.p_states)
        pred = pred - alpha_hat*r - beta_hat # shape: batch_size X size 

        self.modeified_pred = pred    

        return nn.MSELoss()(pred, psn_dev)

    def _offset_loss(self, y, pred, psn_dev):
        beta_hat = (self.lhs_mat @ y).T # Gives the average of y
        # Eliminate ambiguity
        pred = pred - beta_hat # shape: batch_size X size

        self.modeified_pred = pred    
        
        return nn.MSELoss()(pred, psn_dev)
    
    def _absolute_loss(self, y, pred, psn_dev):   
        self.modeified_pred = pred 
        
        return nn.MSELoss()(pred, psn_dev)



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
    def __init__(self, filename, path="/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/"):
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