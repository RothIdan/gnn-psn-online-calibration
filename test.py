import torch
from torch import nn
# from torch.utils.data import DataLoader # Iterable which helps to treat data

import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
from dgl.dataloading import GraphDataLoader
import argparse
from datetime import datetime

from utils import GraphDataset, LossFn
from models import GraphNeuralNetwork, test




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    parser.add_argument('-loss', '--loss_mode', type=str, choices=['absolute', 'offset', 'affine'] , help="loss function error mode", required=True)
    parser.add_argument('-dev', '--deviation', type=int, help="Uniform distributed devation in the range +-'dev' degrees, or the corresponded Normal distribution with the dame std", required=True)
    parser.add_argument('-model', '--model_name', type=str, help="name of the model parameters")

    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    
    dirc = "<full_path_name>" # NOTE: need to replace with the full path name of the directory which contain the script
    datapath = os.path.join(dirc + "data/", args.data_filename)
    modelpath = os.path.join(dirc + "models/", args.model_name)
    model_dict = torch.load(modelpath)

    data_filename = args.data_filename  

    test_data = GraphDataset(data_filename+'_test', datapath)

    # hyper-parameters
    batch_size = model_dict['batch_size']
    d_h = model_dict['h_dim']
    mlp_layer = model_dict['mlp_layer']
    conv_layer = model_dict['conv_layer']
    dropout = model_dict['dropout']

   
    d_in = {'user'    : 2*model_dict['num_meas']*model_dict['num_rfchain'],
            'antenna' : 2*model_dict['num_meas']*model_dict['num_rfchain']}
    
    d_out = model_dict['num_rfchain']
    
    # if args.tx_flag:
    #     d_in = {'user'    : 2*model_dict['num_meas'],
    #             'antenna' : 2*model_dict['num_meas']*model_dict['num_rfchain']}
    # else:
    #     d_in = {'user'    : 2*model_dict['num_meas']*model_dict['num_rfchain'],
    #             'antenna' : 2*model_dict['num_meas']*model_dict['num_rfchain']}

    # d_out = model_dict['num_rfchain']*test_data.Nb if args.tensor_flag else model_dict['num_rfchain']
    
    
    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    model_test = GraphNeuralNetwork(d_in, d_h, d_out, conv_layer, mlp_layer, activation_fn=nn.LeakyReLU(0.1), dropout=dropout, aggr_fn='mean', dev=args.deviation).to(device)
    model_test.load_state_dict(model_dict['model_state_dict'])

    loss_fn = LossFn(args.loss_mode, test_data.Nr, device, test_data.Nb)

    test(test_dataloader, model_test, loss_fn, device)

    print("Done!")
 

   