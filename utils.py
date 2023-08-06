import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
# from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
import numpy as np



# def complex_loss(pilots, combiner, channel, pred):
#     batch_size, Nt, Nrf = pred.shape
#     lhs = torch.complex(pilots[:,:Nrf,:], pilots[:,Nrf:,:])
#     # pilots.view(batch_size, -1) # dtype: float32
#     rhs = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2), combiner), channel) # dtype: complex64
#     # Matrix factorization loss
#     return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))


class LossFn:
    """
    Loss function class.
    Params:
        mod (str): 'sys' - system model loss, 
                   'dev' - phase shifter network deviation loss, 
                   'comb' - their combination.
        alpha (float): the magnitude of effect for the 'dev' loss part in mod 'comb'.
    """

    def __init__(self, mod, alpha=1):
        if mod not in ['sys', 'dev', 'comb']:
            raise KeyError(f'Loss type {mod} not supported.\
                             Use "sys", "dev" or "comb" only')
        else:    
            self.mod = mod
        self.alpha = alpha
        self.counter = 0

    def loss(self, pilots, combiner, channel, pred, psn_dev):
        if self.mod == 'dev':
            return self.__deviation_loss(pred, psn_dev)
        else:
            y_pred = torch.matmul(torch.mul(pred, combiner), channel) # dtype: complex64
            if self.mod == 'sys':
                return self.__system_model_loss(pilots, y_pred) 
            else: # mod == 'comb'
                return self.__system_model_loss(pilots, y_pred) + self.alpha * self.__deviation_loss(pred, psn_dev)


    def __system_model_loss(self, lhs, rhs):
        # Can try also matrix factorization loss - torch.mean(torch.norm(lhs - rhs, dim = (1,2))**2)
        return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))
    
    def __deviation_loss(self, pred, psn_dev):
        batch_size = pred.shape[0]
        pred = pred.angle().view(batch_size, -1)
        psn_dev = psn_dev.angle().view(batch_size, -1)
        return torch.mean(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1])
    
    # def __deviation_loss(self, pred, psn_dev):
    #     return torch.mean((torch.norm(pred - psn_dev, dim=(1,2))**2) / (2*pred[0].numel()))
    

def system_model_loss(lhs, rhs):
        # Can try also matrix factorization loss - torch.mean(torch.norm(lhs - rhs, dim = (1,2))**2)
        return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))
    
def deviation_loss(pred, psn_dev):
        batch_size = pred.shape[0]
        pred = pred.angle().view(batch_size, -1)
        psn_dev = psn_dev.angle().view(batch_size, -1)
        return torch.mean(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1])   # Which would be identical as using nn.MSELoss()(pred, psn_dev)


def rmse(pred, psn_dev):
    batch_size = pred.shape[0]
    pred = pred.angle().rad2deg().view(batch_size, -1)
    psn_dev = psn_dev.angle().rad2deg().view(batch_size, -1)
    
    # pred = torch.ones(pred.shape).to(torch.cuda.current_device()) * 0 # check when using mean 
    
    rmse = torch.sqrt(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1]) # RMSE calculation of every output of the batch

    return rmse.sum() 



def save_model(epochs, model, optimizer, loss_val, config, outfile):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss,
                'best_loss': loss_val,
                'batch_size': config['batch_size'],
                'h_dim': config['h_dim'],
                'num_rfchain': config['num_rfchain'],
                'num_meas': config['num_meas'],
                'mlp_layer': config['mlp_layer'],
                }, outfile)
    
    
# class GraphDataset(Dataset):
#     def __init__(self, data_filename, dataset_path="/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/"):
#         super().__init__(name="graph")
#         self.graphs = dgl.load_graphs(os.path.join(dataset_path, data_filename +".dgl"))
#         data_dict = np.load(os.path.join(dataset_path, data_filename +".npy"))
#         self.psn_dev = data_dict['labels'] # PSN deviation matrices
#         self.pilots = data_dict['features']
#         self.channel = data_dict['channel']
#         self.combiner = data_dict['combiner']


#     def __getitem__(self, i):
#         return self.graphs[i], self.pilots[i], self.combiner[i], self.channel[i], self.psn_dev[i]

#     def __len__(self):
#         return len(self.graphs)
    


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
        try:
            self.M = data_dict['num_meas']
        except:
            self.M = 1
        # self.Nt = data_dict['num_antenna']
        # self.Nue = data_dict['num_user']
        
    def __getitem__(self, i):
        return self.graphs[i], self.pilots[i], self.combiner[i], self.channel[i], self.psn_dev[i]

    def __len__(self):
        return len(self.graphs)
    
    
class GraphDatasetFixed(DGLDataset):
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
        try:
            self.M = data_dict['num_meas']
        except:
            self.M = 1
        # self.Nt = data_dict['num_antenna']
        # self.Nue = data_dict['num_user']
        
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
    

