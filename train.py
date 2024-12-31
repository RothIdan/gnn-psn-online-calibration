import torch
from torch import nn
# from torch.utils.data import DataLoader # Iterable which helps to treat data

import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
import argparse
from datetime import datetime

from utils import GraphDataset, LossFn, save_testset
from models import GraphNeuralNetwork, train, test






def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    parser.add_argument('-loss', '--loss_mode', type=str, choices=['absolute', 'offset', 'affine'] , help="loss function error mode", required=True)
    parser.add_argument('-dev', '--deviation', type=int, help="Uniform distributed devation in the range +-'dev' degrees, or the corresponded Normal distribution with the dame std", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    parser.add_argument('-opt', '--optimizer', type=str, help=" optimizaer type: 'ADAM' or 'SGD' ", default='ADAM')
    parser.add_argument('-tensor', '--tensor_flag', help="activate for the tensor version of phase shifter network deviations", action="store_true")
    parser.add_argument('-tx', '--tx_flag', help="activate for the transmitter calibartion system model version", action="store_true")
    parser.add_argument('-act', '--activation', type=str, help=" activation function: 'ReLU', 'LeakyReLU' or 'ELU' ", default='LeakyReLU')
    parser.add_argument('-mod', '--mode', type=str, help=" wandb operation mode, use 'disabled' for debugging ", default='online')
    # parser.add_argument('-csi', '--channel', help=" when flag is active, CSI would be used in the GNN ", action='store_true')


    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    
    # hyper-parameters
    num_epochs = 200 
    batch_size =  1024 
    learning_rate = 0.0005
    betas = (0.9, 0.999)
    weight_decay = 1e-6
    dropout = 0.3323
    d_h = 1181
    conv_layers = 2
    mlp_layers = 1
    
    
    dirc = "<full_path_name>" # NOTE: need to replace with the full path name of the directory which contain the script
    path = os.path.join(dirc + "data/", args.data_filename)

    dataset = GraphDataset(args.data_filename, path)

    train_data, validation_data, test_data = dgl.data.utils.split_dataset(dataset, [0.88, 0.08, 0.04], random_state=2)#, shuffle=True)

    if not os.path.isfile(path + "/test_dataset"):
        save_testset(path, args.data_filename, test_data, dataset)
   
   # Input and output dimensions based on system version (transmitter/receiver calibration and tensor/non-tensor version)
    if args.tx_flag:
        d_in = {'user'    : 2*dataset.Q,
                'antenna' : 2*dataset.Q*dataset.Nrf}
    else:
        d_in = {'user'    : 2*dataset.Q*dataset.Nrf,
                'antenna' : 2*dataset.Q*dataset.Nrf}

    d_out = dataset.Nrf*dataset.Nb if args.tensor_flag else dataset.Nrf


    if args.activation not in ['ReLU', 'LeakyReLU', 'ELU']:
            raise KeyError(f'Optimizer type {args.activation} not supported. '\
                            'Use "ReLU" or "ELU" only')
    elif args.activation == 'ReLU':
        activation_fn = nn.ReLU()
    elif args.activation == 'LeakyReLU':
        activation_fn = nn.LeakyReLU(0.1)
    elif args.activation == 'ELU':
        activation_fn = nn.ELU(0.1)
    
    loss_fn = LossFn(args.loss_mode, dataset.Nr, device, dataset.Nb)
        
        
    model = GraphNeuralNetwork(d_in, d_h, d_out, conv_layers, mlp_layers, activation_fn=activation_fn, dropout=dropout, aggr_fn='mean', dev=args.deviation).to(device)

    # Create data loaders.
    train_dataloader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    
    print(model)
    
    
    if args.optimizer not in ['ADAM', 'SGD', 'Adagrad', 'Adadelta']:
            raise KeyError(f'Optimizaer type {args.optim} not supported.\
                             Use "ADAM", "SGD", "Adagrad" or "Adadelta" only')
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif args.optimizer == 'SGD':
        momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    

    config = {"graph": args.data_filename, "epochs": num_epochs, "h_dim": d_h, "conv_layer":conv_layers, "mlp_layer": mlp_layers, "batch_size": batch_size, "lr": learning_rate, "num_states": dataset.Nb,
              "weight_decay": weight_decay, "dropout": dropout, "optim": args.optimizer, "num_rfchain": dataset.Nrf, "num_meas": dataset.Q, 'act': args.activation, 'dev':args.deviation, 'error':args.loss_mode} # for W&B 
    if not args.saving_filename: 
        saving_filename = f"{config['graph']}_{datetime.now().strftime('%d-%m-%Y-@-%H:%M')}_{config['error']}_{config['optim']}_conv-layer_{conv_layers}_mlp-layer_" \
                        + f"{mlp_layers}_lr_{learning_rate}_dh_{d_h}_batch_{batch_size}_act_{config['act']}_do_{config['dropout']}_wd_{config['weight_decay']}.pth"
    else:
        saving_filename = args.saving_filename  


    train(train_dataloader, validation_dataloader, model, optimizer, loss_fn, device, num_epochs, config, saving_filename, args.mode)
    
    # Test phase
    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    model_test = GraphNeuralNetwork(d_in, d_h, d_out, conv_layers, mlp_layers, activation_fn=activation_fn, dropout=dropout, aggr_fn='mean', dev=args.deviation).to(device)
    
    path = os.path.join(dirc + "models/", saving_filename)
    model_test.load_state_dict(torch.load(path)['model_state_dict'])
    
    test(test_dataloader, model_test, loss_fn, device)
    
    print("Done!")
 

   