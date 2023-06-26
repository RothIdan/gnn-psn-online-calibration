import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
# from torch.utils.data import Dataset
import torch
import glob
import pandas as pd
import numpy as np



def rmse(pred, psn_dev):
    batch_size = pred.shape[0]
    print(pred.reshape(batch_size, -1)[:4,:8])
    print(psn_dev.reshape(batch_size, -1)[:4,:8])
    pred = pred.angle().rad2deg().view(batch_size, -1)
    psn_dev = psn_dev.angle().rad2deg().view(batch_size, -1)
    # print(pred[:4,:8])
    # print(psn_dev[:4,:8])
    rmse = torch.sqrt(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1]) # RMSE calculation of every output of the batch
    # print(rmse.sum()/batch_size)
    # exit(1)
    return rmse.sum() 



def save_model(epochs, model, optimizer, loss, loss_val, config, outfile):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'best_loss': loss_val,
                # 'batch_size': config['batch_size'],
                # 'h_dim': config['h_dim'],
                # 'num_rf': config['n_rf'],
                # 'mlp_layer': config['mlp_layer'],
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
        self.num_rf = data_dict['rfchain']
        # self.num_t = data_dict['antenna']
        # self.num_ue = data_dict['user']
        
    def __getitem__(self, i):
        return self.graphs[i], self.pilots[i], self.combiner[i], self.channel[i], self.psn_dev[i]

    def __len__(self):
        return len(self.graphs)
    

