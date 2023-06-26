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
from models import GraphNeuralNetwork, GraphNeuralNetwork2, train, test




def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('-x', '--input_filename', type=str, help=".npy file name of radiation pattern data, e.g., 'rad1_train.npy'", required=True)
    # parser.add_argument('-y', '--label_filename', type=str, help=".npy file name of phase data", required=True)
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    # parser.add_argument('-mod', '--model_type', type=str, choices=['vanilla', 'dropout', 'transformer'] , help="model type can be 'vanilla'/'dropout'/'transformer'", required=True)
    # parser.add_argument('-loss', '--loss_type', type=str, choices=['mse', 'myloss'] , help="loss function type can be 'mse'/'myloss'", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    # parser.add_argument('--m', type=int, help="final codebook size", required=True)
    # parser.add_argument('--plot', type=bool, help="plotting the ongoing results", default=False)
    # parser.add_argument('--params', nargs='*', help="list of necessary parameters for the prob. dist.: [mu, sig] for gaussian/...")

    return parser.parse_args()



if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the inti args from input

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    
    # hyper-parameters
    num_epochs = 60
    batch_size = 512
    learning_rate = 1e-4
    betas = (0.9, 0.999)
    weight_decay = 0
    dropout = 0 
    d_h = 128
    n_layer = 3
    
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    path = os.path.join(dirc + "data/", args.data_filename)

    dataset = GraphDataset(args.data_filename, path)
    train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    # Save test dataset for later use
    torch.save(test_data, path + "/test_dataset2")

    d_in = d_out = 2*dataset.num_rf

    # model = GraphNeuralNetwork(d_in, d_h, d_out, activation_fn=nn.ReLU()).to(device) 
    model = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=nn.ReLU()).to(device)

    # Create data loaders.
    train_dataloader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    print(model)
    # print(f"Hyperparameters: lr = {learning_rate}, wd = {weight_decay}, do = {dropout}")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)

    config = {"epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM"}
    # config = {"epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM", "n_rf": dataset.num_rf, "n_t": dataset.num_t, "n_ue": dataset.num_ue} # for W&B
    if not args.saving_filename:
        saving_filename = f"{args.data_filename}_{datetime.now().strftime('%d-%m-%Y-@-%H:%M')}_ADAM_mlp-layer_{n_layer}_lr_{learning_rate}_dh_{d_h}_batch_{batch_size}.pth"
    else:
        saving_filename = args.saving_filename  

    train(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device, num_epochs, config, saving_filename)

    # Test phase
    test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False)
    model_test = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=nn.ReLU()).to(device)
    path = os.path.join(dirc + "models/", saving_filename)
    model_test.load_state_dict(torch.load(path)['model_state_dict'])

    test(test_dataloader, model_test, device)
    print("Done!")
 

   