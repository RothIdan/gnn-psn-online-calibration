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

from utils import GraphDataset, LossFn, GraphDatasetFixed
# from models import GraphNeuralNetwork, GraphNeuralNetwork2, GraphNeuralNetworkDGL, GraphNeuralNetworkDrop, GraphNeuralNetworkConcat2, GraphNeuralNetworkDrop2, WeightedGraphNeuralNetwork,  ChannelGraphNeuralNetwork, GraphAttentionlNetworkDGL, GraphNeuralNetworkConcat, train, test
from models import GraphNeuralNetworkConcat2, GraphNeuralNetworkConcat, train, test, tensor_train, tensor_test

import wandb
import yaml




def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-x', '--input_filename', type=str, help=".npy file name of radiation pattern data, e.g., 'rad1_train.npy'", required=True)
    # parser.add_argument('-y', '--label_filename', type=str, help=".npy file name of phase data", required=True)
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    # parser.add_argument('-mod', '--model_type', type=str, choices=['vanilla', 'dropout', 'transformer'] , help="model type can be 'vanilla'/'dropout'/'transformer'", required=True)
    # parser.add_argument('-loss', '--loss_type', type=str, choices=['mse', 'myloss'] , help="loss function type can be 'mse'/'myloss'", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    parser.add_argument('-opt', '--optimizer', type=str, help=" optimizaer type: 'ADAM' or 'SGD' ", default='ADAM')
    parser.add_argument('-act', '--activation', type=str, help=" activation function: 'ReLU' or 'ELU' ", default='ReLU')
    parser.add_argument('-mod', '--mode', type=str, help=" wandb operation mode, use 'disabled' for debugging ", default='online')
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

    
    # hyper-parameters
    num_epochs = 550
    batch_size =  1024 #512 
    learning_rate = 1e-4  
    betas = (0.9, 0.999)
    weight_decay = 1e-7 #1e-5 #0
    dropout = 0.05 #0.1 #0.26 
    d_h = 1024 #1426 #512 #1321 #512 #928
    n_layer = 1 #1 #2
    
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    path = os.path.join(dirc + "data/", args.data_filename)

    if 'fixed' in os.path.splitext(args.data_filename)[0].split('_'):
        dataset = GraphDatasetFixed(args.data_filename, path)
    else:
        dataset = GraphDataset(args.data_filename, path)

    if args.data_filename == 'graph_stack2':
        train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.92, 0.04, 0.04])
    else:
        # train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
        train_data, validation_data, test_data = dgl.data.utils.split_dataset(dataset, [0.88, 0.08, 0.04], random_state=2)#, shuffle=True)
    # Save test dataset for later use
    if not os.path.isfile(path + "/test_dataset"):
        torch.save(test_data, path + "/test_dataset")
    

    # d_in = d_out = 2*dataset.Nrf
    d_in = 2*dataset.M*dataset.Nrf
    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
        d_out = 2*dataset.Nrf*dataset.Nb
    else:
        d_out = 2*dataset.Nrf
    # d_out = 2*dataset.Nrf*16

    if args.activation not in ['ReLU', 'ELU']:
            raise KeyError(f'Optimizer type {args.activation} not supported.\
                             Use "ReLU" or "ELU" only')
    elif args.activation == 'ReLU':
        activation_fn = nn.ReLU()
    elif args.activation == 'ELU':
        activation_fn = nn.ELU()
        

    # model = GraphNeuralNetwork(d_in, d_h, d_out, activation_fn=nn.ReLU()).to(device) 
    # model = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=activation_fn).to(device)

    # model = ChannelGraphNeuralNetwork(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    # model = GraphNeuralNetworkDrop(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    # model = GraphNeuralNetworkDrop2(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    if '2feat' in os.path.splitext(args.data_filename)[0].split('_'):
        model = GraphNeuralNetworkConcat2(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    else:
        model = GraphNeuralNetworkConcat(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    
    # model = GraphAttentionlNetworkDGL(d_in, d_h, d_out, n_heads=3, activation_fn=activation_fn, dropout=dropout).to(device)
    # model = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=nn.ELU()).to(device)
    # model = WeightedGraphNeuralNetwork(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)

    # Create data loaders.
    train_dataloader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=4)
    
    print(model)
    # print(f"Hyperparameters: lr = {learning_rate}, wd = {weight_decay}, do = {dropout}")
    
    # loss_fn = nn.MSELoss()
    # loss_fn = LossFn(mod='sys')
    
    if args.optimizer not in ['ADAM', 'SGD', 'Adagrad', 'Adadelta']:
            raise KeyError(f'Optimizaer type {args.optim} not supported.\
                             Use "ADAM", "SGD", "Adagrad" or "Adadelta" only')
    elif args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    elif args.optimizer == 'SGD':
        momentum = 0.9
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    

    # config = {"epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM"}
    config = {"graph": args.data_filename, "epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate,
              "weight_decay": weight_decay, "dropout": dropout, "optim": args.optimizer, "num_rfchain": dataset.Nrf, "num_meas": dataset.M, 'act': args.activation} # for W&B 
    if not args.saving_filename: 
        saving_filename = f"{config['graph']}_{datetime.now().strftime('%d-%m-%Y-@-%H:%M')}_{config['optim']}_mlp-layer_" \
                        + f"{n_layer}_lr_{learning_rate}_dh_{d_h}_batch_{batch_size}_act_{config['act']}_do_{config['dropout']}_wd_{config['weight_decay']}.pth"
    else:
        saving_filename = args.saving_filename  

    # train(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device, num_epochs, config, saving_filename)
    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):    
        tensor_train(train_dataloader, validation_dataloader, model, optimizer, device, num_epochs, config, saving_filename, args.mode)
    else:
        train(train_dataloader, validation_dataloader, model, optimizer, device, num_epochs, config, saving_filename, args.mode)
    # Test phase
    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False)
    # model_test = GraphNeuralNetworkDrop(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    model_test = GraphNeuralNetworkConcat(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    if '2feat' in os.path.splitext(args.data_filename)[0].split('_'):
        model_test = GraphNeuralNetworkConcat2(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    path = os.path.join(dirc + "models/", saving_filename)
    model_test.load_state_dict(torch.load(path)['model_state_dict'])

    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
        tensor_test(test_dataloader, model_test, device)
    else:
        test(test_dataloader, model_test, device)
    print("Done!")
 

   