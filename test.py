import torch
from torch import nn
# from torch.utils.data import DataLoader # Iterable which helps to treat data

import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
from dgl.dataloading import GraphDataLoader
import argparse
from datetime import datetime

from utils import GraphDataset
from models import GraphNeuralNetwork, GraphNeuralNetwork2, GraphNeuralNetworkDGL, GraphNeuralNetworkDrop, train, test




def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-x', '--input_filename', type=str, help=".npy file name of radiation pattern data, e.g., 'rad1_train.npy'", required=True)
    # parser.add_argument('-y', '--label_filename', type=str, help=".npy file name of phase data", required=True)
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    # parser.add_argument('-mod', '--model_type', type=str, choices=['vanilla', 'dropout', 'transformer'] , help="model type can be 'vanilla'/'dropout'/'transformer'", required=True)
    # parser.add_argument('-loss', '--loss_type', type=str, choices=['mse', 'myloss'] , help="loss function type can be 'mse'/'myloss'", required=True)
    parser.add_argument('-model', '--model_name', type=str, help="name of the model parameters")
    # parser.add_argument('--m', type=int, help="final codebook size", required=True)
    # parser.add_argument('--plot', type=bool, help="plotting the ongoing results", default=False)
    # parser.add_argument('--params', nargs='*', help="list of necessary parameters for the prob. dist.: [mu, sig] for gaussian/...")

    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    datapath = os.path.join(dirc + "data/", args.data_filename)
    modelpath = os.path.join(dirc + "models/", args.model_name)
    model_dict = torch.load(modelpath)

    data_filename = args.data_filename  
    test_data = torch.load(datapath + "/test_dataset")

    # hyper-parameters
    batch_size = model_dict['batch_size']
    # weight_decay = 0
    # dropout = 0 
    d_h = model_dict['h_dim']
    n_layer = model_dict['mlp_layer']
    d_in = 2*model_dict['num_meas']*model_dict['num_rfchain']
    d_out = 2*model_dict['num_rfchain']
    dropout = 0.1
    # batch_size = 512
    # d_h = 256
    # n_layer = 2
    # d_in = d_out = 2*2

    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # model = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=nn.ReLU()).to(device)
    model = GraphNeuralNetworkDrop(d_in, d_h, d_out, n_layer, activation_fn=nn.ReLU(), dropout=dropout).to(device)

    model.load_state_dict(model_dict['model_state_dict'])

    test(test_dataloader, model, device)
    print("Done!")
 

   