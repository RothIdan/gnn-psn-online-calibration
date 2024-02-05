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

from utils import GraphDataset, LossFn, GraphDatasetFixed, save_testset
# from models import GraphNeuralNetwork, GraphNeuralNetwork2, GraphNeuralNetworkDGL, GraphNeuralNetworkDrop, GraphNeuralNetworkConcat2, GraphNeuralNetworkDrop2, WeightedGraphNeuralNetwork,  ChannelGraphNeuralNetwork, GraphAttentionlNetworkDGL, GraphNeuralNetworkConcat, train, test
from models import GraphNeuralNetworkConcat2, GraphNeuralNetworkConcat, EdgeGraphNeuralNetwork, SkipConGnn, GraphNeuralNetworkConcatDecision, GraphNeuralNetworkCSIConcat, train, test, tensor_train, tensor_test
from models import Edge3DGNN, edge_train, edge_test
import wandb
import yaml

# from utils import MLPDataset
# from models import EightLayerNet 




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    # parser.add_argument('-loss', '--loss_type', type=str, choices=['mse', 'myloss'] , help="loss function type can be 'mse'/'myloss'", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    parser.add_argument('-opt', '--optimizer', type=str, help=" optimizaer type: 'ADAM' or 'SGD' ", default='ADAM')
    parser.add_argument('-act', '--activation', type=str, help=" activation function: 'ReLU' or 'ELU' ", default='ReLU')
    parser.add_argument('-mod', '--mode', type=str, help=" wandb operation mode, use 'disabled' for debugging ", default='online')
    parser.add_argument('-csi', '--channel', help=" when flag is active, CSI would be used in the GNN ", action='store_true')
    # parser.add_argument('--m', type=int, help="final codebook size", required=True)
    # parser.add_argument('--plot', type=bool, help="plotting the ongoing results", default=False)
    # parser.add_argument('--params', nargs='*', help="list of necessary parameters for the prob. dist.: [mu, sig] for gaussian/...")

    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    
    # hyper-parameters
    num_epochs = 400 
    batch_size =  128 
    learning_rate = 0.0005344
    betas = (0.9, 0.999)
    weight_decay = 1e-7 
    dropout = 0.02405
    d_h = 582
    conv_layers = 3
    mlp_layers = 2 
    
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    path = os.path.join(dirc + "data/", args.data_filename)

    if 'fixed' in os.path.splitext(args.data_filename)[0].split('_'):
        dataset = GraphDatasetFixed(args.data_filename, path)
    else:
        dataset = GraphDataset(args.data_filename, path)

    train_data, validation_data, test_data = dgl.data.utils.split_dataset(dataset, [0.88, 0.08, 0.04], random_state=2)#, shuffle=True)
    # Save test dataset for later use
    if not os.path.isfile(path + "/test_dataset"):
        save_testset(path, args.data_filename, test_data, dataset)
    

    # d_in = {('antenna', 'channel', 'user') : 2,
    #         ('user', 'channel', 'antenna') : 2,
    #         ('rfchain', 'psn', 'antenna')  : 2*dataset.Q,
    #         ('antenna', 'psn', 'rfchain')  : 2*dataset.Q,
    #         ('user', 'pilot', 'rfchain')   : 2*dataset.Q,
    #         ('rfchain', 'pilot', 'user')   : 2*dataset.Q }
    

    # d_in = d_out = 2*dataset.Nrf
    d_in = 2*dataset.Q*dataset.Nrf
    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
        d_out = 2*dataset.Nrf*dataset.Nb
    else:
        d_out = 2*dataset.Nrf
    # d_out = 2*dataset.Nrf*16

    if 'mlp' in os.path.splitext(args.data_filename)[0].split('_'):
        d_in = d_in * (3+8)
        d_out = d_out * 8

    if '3D' in os.path.splitext(args.data_filename)[0].split('_'):
        d_in = {'channel' : 2,
                'psn'     : 2*dataset.Q,
                'pilot'   : 2*dataset.Q }
    
        d_out = 2

    if args.activation not in ['ReLU', 'LeakyReLU', 'ELU']:
            raise KeyError(f'Optimizer type {args.activation} not supported. '\
                            'Use "ReLU" or "ELU" only')
    elif args.activation == 'ReLU':
        activation_fn = nn.ReLU()
    elif args.activation == 'LeakyReLU':
        activation_fn = nn.LeakyReLU(0.1)
    elif args.activation == 'ELU':
        activation_fn = nn.ELU()
        
        
    # if 'channel' in os.path.splitext(args.data_filename)[0].split('_') or args.channel == True:
    #     model = EdgeGraphNeuralNetwork(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    #     # model = GraphNeuralNetworkCSIConcat(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout, Nt=dataset[0][0].num_nodes('antenna')).to(device)
    # elif '2feat' in os.path.splitext(args.data_filename)[0].split('_'):
    #     model = GraphNeuralNetworkConcat2(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    #     # model = SkipConGnn(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    # elif 'decision'in os.path.splitext(args.data_filename)[0].split('_'):
    #     model = GraphNeuralNetworkConcatDecision(d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout, M=dataset.M, Nrf=dataset.Nrf).to(device)
    # else:
    #     model = GraphNeuralNetworkConcat(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    
    # model = EdgeGraphNeuralNetwork(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    model = Edge3DGNN(d_in, d_h, d_out, conv_layers, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)

    # Create data loaders.
    train_dataloader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
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
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif args.optimizer == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    

    # config = {"epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM"}
    config = {"graph": args.data_filename, "epochs": num_epochs, "h_dim": d_h, "conv_layer":conv_layers, "mlp_layer": mlp_layers, "batch_size": batch_size, "lr": learning_rate, "num_states": dataset.Nb,
              "weight_decay": weight_decay, "dropout": dropout, "optim": args.optimizer, "num_rfchain": dataset.Nrf, "num_meas": dataset.Q, 'act': args.activation} # for W&B 
    if not args.saving_filename: 
        saving_filename = f"mlp_aggr_{config['graph']}_{datetime.now().strftime('%d-%m-%Y-@-%H:%M')}_{config['optim']}_conv-layer_{conv_layers}_mlp-layer_" \
                        + f"{mlp_layers}_lr_{learning_rate}_dh_{d_h}_batch_{batch_size}_act_{config['act']}_do_{config['dropout']}_wd_{config['weight_decay']}.pth"
    else:
        saving_filename = args.saving_filename  


    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):    
        tensor_train(train_dataloader, validation_dataloader, model, optimizer, device, num_epochs, config, saving_filename, args.mode)
    elif '3D' in os.path.splitext(args.data_filename)[0].split('_'):
        edge_train(train_dataloader, validation_dataloader, model, optimizer, device, num_epochs, config, saving_filename, args.mode)
    else:
        train(train_dataloader, validation_dataloader, model, optimizer, device, num_epochs, config, saving_filename, args.mode)
    
    
    # Test phase
    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)

    # if 'channel' in os.path.splitext(args.data_filename)[0].split('_')  or args.channel == True:
    #     model_test = EdgeGraphNeuralNetwork(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    #     # model_test = GraphNeuralNetworkCSIConcat(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout, Nt=dataset[0][0].num_nodes('antenna')).to(device)
    # elif '2feat' in os.path.splitext(args.data_filename)[0].split('_'):
    #     model_test = GraphNeuralNetworkConcat2(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    #     # model_test = SkipConGnn(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
    # elif 'decision'in os.path.splitext(args.data_filename)[0].split('_'):
    #     model_test = GraphNeuralNetworkConcatDecision(d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout, M=dataset.M, Nrf=dataset.Nrf).to(device)
    # else:
    #     model_test = GraphNeuralNetworkConcat(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)

    # model_test = EdgeGraphNeuralNetwork(d_in, d_h, d_out, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    model_test = Edge3DGNN(d_in, d_h, d_out, conv_layers, mlp_layers, activation_fn=activation_fn, dropout=dropout).to(device)
    
    path = os.path.join(dirc + "models/", saving_filename)
    model_test.load_state_dict(torch.load(path)['model_state_dict'])
    

    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
        tensor_test(test_dataloader, model_test, device)
    elif '3D' in os.path.splitext(args.data_filename)[0].split('_'):
        edge_test(test_dataloader, model_test, device)
    else:
        test(test_dataloader, model_test, device)
    print("Done!")
 

   