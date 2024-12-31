import torch
from torch import nn
# from torch.utils.data import DataLoader # Iterable which helps to treat data

import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
import argparse

from utils import GraphDataset, system_model_loss, GraphDatasetFixed, deviation_loss, LossFn
from models import GraphNeuralNetworkConcat2, GraphNeuralNetworkConcat4, Edge3DGNN, EdgeGraphNeuralNetwork, EdgeRegressionGNN

import wandb, yaml, pprint, functools
from multiprocessing import Process
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau

import copy



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    parser.add_argument('-loss', '--loss_mode', type=str, choices=['absolute', 'offset', 'affine'] , help="loss function error mode", required=True)
    parser.add_argument('-dev', '--deviation', type=int, help="Uniform distributed devation in the range +-'dev' degrees, or the corresponded Normal distribution with the dame std", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    parser.add_argument('-tensor', '--tensor_flag', help="activate for the tensor version of phase shifter network deviations", action="store_true")
    parser.add_argument('-tx', '--tx_flag', help="activate for the transmitter calibartion system model version", action="store_true")
    # parser.add_argument('-id', '--id', help=" sweep id from wandb", required=True)

    return parser.parse_args()


def train_old(train_data, validation_data, d_in, d_out, M, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        # model = GraphNeuralNetworkConcat2(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        model = EdgeGraphNeuralNetwork(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)

        if config.optim == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            # Train
            cumu_loss = 0  
            for i, (g, _, _, _, psn_dev) in enumerate(train_dataloader):
                # g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                g, psn_dev = g.to(device), psn_dev.to(device)
                # Forward pass
                pred = model(g)
                # y_pred = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2).repeat(1,M,1), combiner), channel) # dtype: complex64


                # loss = system_model_loss(pilots, y_pred) + alpha * deviation_loss(pred, psn_dev)
                # loss = system_model_loss(pilots, y_pred)
                loss = deviation_loss(torch.transpose(pred, dim0=1,dim1=2), psn_dev)
                cumu_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = cumu_loss / len(train_dataloader)

            # Validation
            if (epoch+1) % 3 == 0:
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for g, pilots, combiner, channel, psn_dev in validation_dataloader:
                        # g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                        g, psn_dev = g.to(device), psn_dev.to(device)
                        # Forward pass
                        pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64
                        # y_pred = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2).repeat(1,M,1), combiner), channel) # dtype: complex64
                            
                        # valid_loss += system_model_loss(pilots, y_pred).item()
                        valid_loss += deviation_loss(torch.transpose(pred, dim0=1,dim1=2), psn_dev).item()

                valid_loss /= len(validation_dataloader)
                model.train()
            
                wandb.log({"loss": loss, "validation loss": valid_loss})

            else:
                wandb.log({"loss": loss})



def train(train_data, validation_data, test_data, d_in, d_out, path, loss_fn, dev, device, config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)
        test_dataloader = GraphDataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        if config.activation == 'ReLU':
            activation_fn = nn.ReLU()
        elif config.activation == 'LeakyReLU':
            activation_fn = nn.LeakyReLU(0.1)
        else:
            activation_fn = nn.ELU(0.1)

        # if config.model == "mask":
        #     model = GraphNeuralNetworkConcat2(d_in, config.d_h, d_out, config.conv_layers, config.mlp_layers, activation_fn=activation_fn, dropout=config.dropout).to(device)
        # else:
        model = GraphNeuralNetworkConcat4(d_in, config.d_h, d_out, config.conv_layers, config.mlp_layers, activation_fn=activation_fn, dropout=config.dropout, aggr_fn='mean', dev=dev).to(device)
        # model = GraphNeuralNetworkConcat4(d_in, config.d_h, d_out, config.conv_layers, config.mlp_layers, activation_fn=activation_fn, dropout=config.dropout, aggr_fn=config.aggregation).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum, nesterov=True)
        # scheduler = StepLR(optimizer, step_size=80, gamma=0.3) # every step_size epochs: lr <-- lr * gamma
        # scheduler = MultiStepLR(optimizer, milestones=[80,120,180], gamma=0.3)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        

        # if config.optim == 'ADAM':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # elif config.optim == 'SGD':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
        best_loss = np.inf
        for epoch in range(config.epochs):
            # Train
            cumu_loss = 0  
            # for i, (g, pilots, combiner, channel, psn_dev) in enumerate(train_dataloader):
            #     g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
            for i, (g, _, _, _, psn_dev) in enumerate(train_dataloader):
                g, psn_dev = g.to(device), psn_dev.to(device)
                Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
                Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None

                pred = model(g) # shape: batch X Nt X Nrf (or Nrf*Nb) , dtype: Complex64

                # y_pred = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2).repeat(1,M,1), combiner), channel) # dtype: complex64
                
                psn_dev = psn_dev.angle().view(config.batch_size, -1)
                # psn_dev = psn_dev.angle().view(config.batch_size, -1)[:,1:]
                # psn_dev = psn_dev.angle().view(config.batch_size, -1)[:,1:] - psn_dev.angle().view(config.batch_size, -1)[:,0:1]

                loss = 0
                for w in pred:
                    # # w = w.reshape(config.batch_size, Nrf, Nt) if Nb is None else w.reshape(config.batch_size, Nrf, Nt, Nb) # dtype: Complex64
                    # w = torch.transpose(w, dim0=2, dim1=1) if Nb is None else torch.transpose(w.reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1) # dtype: Complex64
                    # loss += deviation_loss(w, psn_dev)

                    # system model loss
                    # Q = pilots.shape[1] // Nrf
                    # w = torch.transpose(w, dim0=2, dim1=1) if Nb is None else torch.transpose(w.reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1) # dtype: Complex64
                    # lhs = torch.concat((pilots.reshape(config.batch_size, -1).real, pilots.reshape(config.batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
                    # rhs = torch.matmul(torch.mul(torch.tile(w, (1,Q,1)), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
                    # rhs = torch.concat((rhs.reshape(config.batch_size, -1).real, rhs.reshape(config.batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

                    # loss += system_model_loss(lhs, rhs)

                    # Phase only estimation
                    # w = w.reshape(config.batch_size, Nrf, Nt) if Nb is None else w.reshape(config.batch_size, Nrf, Nt, Nb) # dtype: Complex64

                    w = torch.transpose(w, dim0=2, dim1=1) if Nb is None else torch.transpose(w.reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1) # dtype: Complex64
                    # w = w.reshape(config.batch_size, -1)[:,1:]
                    
                    w = w.reshape(config.batch_size, -1)
                    # w = w[:,1:] - w[:,0:1]
                    
                    loss += loss_fn.loss(w, psn_dev)
                    # loss += nn.MSELoss()(w, psn_dev)

                    # m = torch.minimum(torch.abs(w - psn_dev), 2*np.pi - torch.abs(w - psn_dev)) # element-wise minimum
                    # loss += torch.mean(m**2)

                cumu_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = cumu_loss / len(train_dataloader)
            
            # Validation
            if (epoch+1) % 3 == 0:
                model.eval()
                valid_loss, valid_rmse = 0, 0
                with torch.no_grad():
                    # for g, pilots, combiner, channel, psn_dev in validation_dataloader:
                    #     g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                    for g, _, _, _, psn_dev in validation_dataloader:
                        g, psn_dev = g.to(device), psn_dev.to(device)
                        Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
                        Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
                                                
                        pred = model(g) # shape: batch X Nt-1 X N_rf * Nb (out_feats), dtype: Complex64
                        # # pred = pred[-1].reshape(config.batch_size, Nrf, Nt) if Nb is None else pred[-1].reshape(config.batch_size, Nrf, Nt, Nb)
                        # pred = torch.transpose(pred[-1], dim0=2, dim1=1) if Nb is None else torch.transpose(pred[-1].reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1) 
               
                        # valid_loss += deviation_loss(pred, psn_dev).item()
                        # valid_rmse += rmse(pred, psn_dev)

                        # system model loss
                        # Q = pilots.shape[1] // Nrf
                        # w = torch.transpose(pred[-1], dim0=2, dim1=1) if Nb is None else torch.transpose(pred[-1].reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1)
                        # lhs = torch.concat((pilots.reshape(config.batch_size, -1).real, pilots.reshape(config.batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
                        # rhs = torch.matmul(torch.mul(torch.tile(w, (1,Q,1)), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
                        # rhs = torch.concat((rhs.reshape(config.batch_size, -1).real, rhs.reshape(config.batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

                        # valid_loss += system_model_loss(lhs, rhs)
                        # valid_rmse += rmse(w, psn_dev)

                        # Phase only estimation
                        # pred = pred[-1].reshape(config.batch_size, Nrf, Nt) if Nb is None else pred[-1].reshape(config.batch_size, Nrf, Nt, Nb)

                        pred = torch.transpose(pred[-1], dim0=2, dim1=1) if Nb is None else torch.transpose(pred[-1].reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1)
                        pred = pred.reshape(config.batch_size, -1)
                        # pred = pred.reshape(config.batch_size, -1)[:,1:]
                        # pred = pred[-1].reshape(config.batch_size, -1)
                        # pred = pred[:,1:] - pred[:,0:1]

                        # psn_dev = psn_dev.angle().view(config.batch_size, -1)[:,1:]
                        # psn_dev = psn_dev.angle().view(config.batch_size, -1)[:,1:] - psn_dev.angle().view(config.batch_size, -1)[:,0:1]
                        psn_dev = psn_dev.angle().view(config.batch_size, -1)

                        valid_loss += loss_fn.loss(pred, psn_dev).item()
                        # valid_loss += nn.MSELoss()(pred, psn_dev).item()
                        # valid_rmse += torch.sqrt(torch.mean((pred.rad2deg()-psn_dev.rad2deg())**2, dim=1)).sum() 

                        # valid_rmse += torch.mean((pred.rad2deg() - psn_dev.rad2deg())**2, dim=1).sum()
                        pred = loss_fn.get_modified_pred()
                        valid_rmse += torch.mean((pred - psn_dev)**2, dim=1).sum()

                        # m = torch.minimum(torch.abs(pred - psn_dev), 2*np.pi - torch.abs(pred - psn_dev)) # element-wise minimum
                        # valid_loss += torch.mean(m**2).item()
                        # m = torch.minimum(torch.abs(pred.rad2deg() - psn_dev.rad2deg()), 360 - torch.abs(pred.rad2deg() - psn_dev.rad2deg())) # element-wise minimum
                        # valid_rmse += torch.sqrt(torch.mean(m**2, dim=1)).sum()
                        

                valid_loss /= len(validation_dataloader)
                valid_rmse = torch.sqrt(valid_rmse / len(validation_dataloader.dataset))

                scheduler.step(valid_loss)

                model.train()
            
                wandb.log({"loss": loss, "validation loss": valid_loss, "validation RMSE" : valid_rmse.rad2deg()})

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = copy.deepcopy(model)
                    counter = 0
                else:
                    counter += 1
                    if counter == 13: # early stopping after 39 epochs of no imporvement
                        break

            else:
                wandb.log({"loss": loss})
            
            if (epoch+1) == 60 and valid_rmse > (dev / (3**0.5)): # Early stopping
            # if (epoch+1) == 60 and valid_rmse > 8.5: # Early stopping
                    break
            
            # scheduler.step()
        try:
            torch.save({'epoch': epoch+1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(path, f'/models/{wandb.run.name}.pth'))
        except:
            print("Not Saved")
        # Test best model
        best_model.eval()
        # test_loss = 0
        # mse = 0
        test_rmse = 0

        with torch.no_grad():
            for g, _, _, _, psn_dev in test_dataloader:
                g, psn_dev = g.to(device), psn_dev.to(device)
                Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
                Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
                
                # Forward pass
                pred = best_model(g) # shape: batch X N_t X N_rf (out_feats/2), type: Complex
                pred = torch.transpose(pred[-1], dim0=2, dim1=1) if Nb is None else torch.transpose(pred[-1].reshape(config.batch_size, Nt, Nrf, Nb), dim0=2, dim1=1)

                # Phase only estimation
                # pred = pred.reshape(config.batch_size, -1)[:,1:] # We care about the deviations difference only
                # psn_dev = psn_dev.angle().view(config.batch_size, -1)[:,1:]
                pred = pred.reshape(config.batch_size, -1)
                psn_dev = psn_dev.angle().view(config.batch_size, -1)

                _ = loss_fn.loss(pred, psn_dev).item() # Just to modify pred for rmse calculation
                pred = loss_fn.get_modified_pred()
                # test_loss += nn.MSELoss()(pred, psn_dev).item()
                test_rmse += torch.mean((pred-psn_dev)**2, dim=1).sum()
                # mse += torch.sum((pred-psn_dev)**2, dim=1).sum()
    
        test_rmse = torch.sqrt(test_rmse / len(test_dataloader.dataset))
        # test_loss /= num_batches
        wandb.log({"test rmse": test_rmse.rad2deg()})

            


def tensor_train(train_data, validation_data, d_in, d_out, M, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)


        # model = GraphNeuralNetworkConcat2(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        model = EdgeGraphNeuralNetwork(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)

        if config.optim == 'ADAM':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        elif config.optim == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

        for epoch in range(config.epochs):
            # Train
            cumu_loss = 0  
            for i, (g, _, _, _, psn_dev) in enumerate(train_dataloader):
                # g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                g, psn_dev = g.to(device), psn_dev.to(device)
                _, Nrf, Nt, Nb = psn_dev.shape
                pred = model(g) # shape: batch X N_t X N_rf * Nb (out_feats/2), dtype: Complex64
                # pred = torch.transpose(pred.reshape(config.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2)
                # loss = deviation_loss(pred, psn_dev)


                # y_pred = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2).repeat(1,M,1), combiner), channel) # dtype: complex64

                # loss = system_model_loss(pilots, y_pred) + alpha * deviation_loss(pred, psn_dev)
                # loss = system_model_loss(pilots, y_pred)

                
                loss = 0
                for w in pred:
                    w = torch.transpose(w.reshape(config.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2) # shape: batch X N_rf X N_t X N_b
                    loss += deviation_loss(w, psn_dev)

                cumu_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = cumu_loss / len(train_dataloader)
            
            # Validation
            if (epoch+1) % 3 == 0:
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for g, _, _, _, psn_dev in validation_dataloader:
                        # g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                        g, psn_dev = g.to(device), psn_dev.to(device)
                        _, Nrf, Nt, Nb = psn_dev.shape
                        pred = model(g) # shape: batch X N_t X N_rf * Nb (out_feats/2), dtype: Complex64
                        pred = torch.transpose(pred[-1].reshape(config.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2)
                        # valid_loss += system_model_loss(pilots, y_pred).item()
                        valid_loss += deviation_loss(pred, psn_dev).item()

                valid_loss /= len(validation_dataloader)

                model.train()
            
                wandb.log({"loss": loss, "validation loss": valid_loss})

            else:
                wandb.log({"loss": loss})
            
            if (epoch+1) == 150 and valid_loss > 0.012: # Early stopping
                    break



def edge_train(train_data, validation_data, d_in, d_out, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        if config.activation == 'ReLU':
            activation_fn = nn.ReLU()
        elif config.activation == 'LeakyReLU':
            activation_fn = nn.LeakyReLU(0.1)
        else:
            activation_fn = nn.SELU()

        model = Edge3DGNN(d_in, config.d_h, d_out, config.conv_layers, config.mlp_layers, activation_fn=activation_fn, dropout=config.dropout).to(device)

        # if config.optim == 'ADAM':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # elif config.optim == 'SGD':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        batch_size = train_dataloader.batch_size
        scheduler = StepLR(optimizer, step_size=100, gamma=0.3) # every step_size epochs: lr <-- lr * gamma
        

        for epoch in range(config.epochs):
            # Train
            cumu_loss = 0
            for i, (g, pilots, combiner, channel, psn_dev) in enumerate(train_dataloader):  
                # Each i is a batch of batch_size samples
                # Shapes: psn_dev: batch X Nrf X Nt (X Nb), channel: batch X Nt X Nue, pilots: batch X Q*Nrf X Nue, combiner: Q*Nrf X Nt
                g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                
                Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
                Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
                Q = pilots.shape[1] // Nrf

                # Predict the PSN deviation matrices - W
                pred = model(g) # shape: batch X Nt*Nrf X 1 or Nb, dtype: Complex64 

                # Process data for loss calculation
                loss = 0
                for w in pred:
                    w = w.reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64
                    lhs = torch.concat((pilots.reshape(batch_size, -1).real, pilots.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
                    rhs = torch.matmul(torch.mul(torch.kron(torch.ones(Q,1).to(device), w), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
                    rhs = torch.concat((rhs.reshape(batch_size, -1).real, rhs.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

                    loss += system_model_loss(lhs, rhs)
            
                cumu_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss = cumu_loss / len(train_dataloader)
            scheduler.step()

            # validation step
            if (epoch+1) % 3 == 0:
                # valid_loss = validate(validloader, model, loss_fn, device, dataloader.dataset.max, dataloader.dataset.min)
                num_batches = len(validation_dataloader)
                size = len(validation_dataloader.dataset) 
                model.eval()
                valid_loss = 0
                with torch.no_grad():
                    for g, pilots, combiner, channel, psn_dev in validation_dataloader:
                
                        g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                        
                        pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64

                        w = pred[-1].reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64
                        lhs = torch.concat((pilots.reshape(batch_size, -1).real, pilots.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
                        rhs = torch.matmul(torch.mul(torch.kron(torch.ones(Q,1).to(device), w), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
                        rhs = torch.concat((rhs.reshape(batch_size, -1).real, rhs.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

                        valid_loss += system_model_loss(lhs, rhs).item()
                      
                
                valid_loss /= num_batches

                model.train()
            
                wandb.log({"loss": loss, "validation loss": valid_loss})

            else:
                wandb.log({"loss": loss})
            
            if (epoch+1) == 60 and valid_loss > 40: # Early stopping
                break

    del model
    torch.cuda.empty_cache()
    


def regressor_train(train_data, validation_data, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        if config.activation == 'ReLU':
            activation_fn = nn.ReLU()
        elif config.activation == 'LeakyReLU':
            activation_fn = nn.LeakyReLU(0.1)
        else:
            activation_fn = nn.ELU(1)

        if len(dataset[0][0].canonical_etypes) == 4 or 6:  
            mod = f'{len(dataset[0][0].canonical_etypes)}rel'
            d_in = {'user'    : 2*dataset.Q*dataset.Nrf,
                    'antenna' : 2*dataset.Q*dataset.Nrf,
                    'rfchain' : 2*dataset.Q*dataset.Nue }
            d_out = 2
        
        else:
            mod = '2rel'
            d_in = {'antenna' : 2*dataset.Q*dataset.Nrf,
                    'rfchain' : 2*dataset.Q*dataset.Nue }
            d_out = 2

        model = EdgeRegressionGNN(mod, d_in, config.d_h, d_out, config.conv_layers, config.mlp_layers, activation_fn=activation_fn, dropout=config.dropout).to(device)

        # if config.optim == 'ADAM':
        #     optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # elif config.optim == 'SGD':
        #     optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        batch_size = train_dataloader.batch_size
        scheduler = StepLR(optimizer, step_size=80, gamma=0.3) # every step_size epochs: lr <-- lr * gamma

        for epoch in range(config.epochs):
            model.train()
            cumu_loss = 0
            for i, (g, _, _, _, psn_dev) in enumerate(train_dataloader):  
                # Each i is a batch of batch_size samples
                # Shapes: psn_dev: batch X Nrf X Nt (X Nb), channel: batch X Nt X Nue, pilots: batch X Q*Nrf X Nue, combiner: Q*Nrf X Nt
                g, psn_dev = g.to(device), psn_dev.to(device)
                
                Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
                Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
                # Q = pilots.shape[1] // Nrf

                # Predict the PSN deviation matrices - W
                pred = model(g) # shape: batch X Nt*Nrf X 1 or Nb, dtype: Complex64 

                # Process data for loss calculation
                loss = 0
                for w in pred:
                    w = w.reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb)
                    loss += deviation_loss(w, psn_dev)
            
                cumu_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            avg_loss = cumu_loss / len(train_dataloader)
            scheduler.step()

            # validation step
            if (epoch+1) % 3 == 0:
                num_batches = len(validation_dataloader)
                model.eval()
                valid_loss = 0

                with torch.no_grad():
                    for g, _, _, _, psn_dev in validation_dataloader:
                        g, psn_dev = g.to(device), psn_dev.to(device)
                        
                        pred = model(g) # shape: batch X Nt*Nrf X 1 or Nb, dtype: Complex64 

                        w = pred[-1].reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64
                        
                        valid_loss += deviation_loss(w, psn_dev).item()
                
                valid_loss /= num_batches

                wandb.log({"loss": loss, "validation loss": valid_loss})

            else:
                wandb.log({"loss": loss})
            
            if (epoch+1) == 50 and valid_loss > 0.009: # Early stopping
                break



WANDB_AGENT_MAX_INITIAL_FAILURES=1000

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    with open('/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/sweep_config.yaml', 'r') as stream:
        sweep_config = yaml.safe_load(stream)
    
    pprint.pprint(sweep_config)
    sweep_id = wandb.sweep(sweep_config, project="gnn_psn_calib") 
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    path = os.path.join(dirc + "data/", args.data_filename)


    dataset = GraphDatasetFixed(args.data_filename, path)

    # train_data, validation_data = dgl.data.utils.split_dataset(dataset, [0.9, 0.1], random_state=2)
    train_data, validation_data, test_data = dgl.data.utils.split_dataset(dataset, [0.88, 0.08, 0.04], random_state=2)
    
    if args.tx_flag:
        d_in = {'user'    : 2*dataset.Q,
                'antenna' : 2*dataset.Q*dataset.Nrf}
    else:
        d_in = {'user'    : 2*dataset.Q*dataset.Nrf,
                'antenna' : 2*dataset.Q*dataset.Nrf}

    d_out = dataset.Nrf*dataset.Nb if args.tensor_flag else dataset.Nrf
  
    loss_fn = LossFn(args.loss_mode, d_out*dataset.Nt, device)


    # d_in = 2*dataset.Q*dataset.Nrf
    # if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
    #     d_out = 2*dataset.Nrf*dataset.Nb
    # else:
    #     # d_out = 2*dataset.Nrf
    #     d_out = dataset.Nrf
    
    # if 'edge' in os.path.splitext(args.data_filename)[0].split('_'):
    #     d_in = {'channel' : 2,
    #             'psn'     : 2*dataset.Q,
    #             'pilot'   : 2*dataset.Q }
        
    #     d_out = 2
    
    
    # device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # train_func1 = functools.partial(train, device1, train_data, validation_data, d_in, d_out, dataset.M)
    # train_func2 = functools.partial(train, device2, train_data, validation_data, d_in, d_out, dataset.M)

    # wandb.agent(sweep_id, function=train_func1, count=50)
    # wandb.agent(sweep_id, function=train_func2, count=50)

    # train_func = functools.partial(train, train_data, validation_data, d_in, d_out)
    train_func = functools.partial(train, train_data, validation_data, test_data, d_in, d_out, path, loss_fn, args.deviation, device)

    # train_func = functools.partial(tensor_train, train_data, validation_data, d_in, d_out, dataset.M)
    # train_func = functools.partial(edge_train, train_data, validation_data, d_in, d_out)
    # train_func = functools.partial(regressor_train, train_data, validation_data)

    # running sweep
    wandb.agent(sweep_id, function=train_func, count=15)
    wandb.finish()

   
   
    # procs = []
    # num_processes = 2
    # for i in range(num_processes):
    #     proc = Process(target=wandb.agent, args=(sweep_id, train_func, None, None, 50))
    #     procs.append(proc)
    #     proc.start()

    # for proc in procs:
    #     proc.join()

    # proc1 = Process(target=wandb.agent, args=(sweep_id, train_func1, None, None, 50))
    # procs.append(proc1)
    # proc1.start()
    # proc2 = Process(target=wandb.agent, args=(sweep_id, train_func2, None, None, 50))
    # procs.append(proc2)
    # proc2.start()
    # for proc in procs:
    #     proc.join()





# if __name__ == "__main__":
#     # torch.autograd.set_detect_anomaly(True)
#     torch.manual_seed(2)
#     np.random.seed(2)

#     args = parse_args() # getting all the init args from input

#     device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"device: {device}")

    
#     # hyper-parameters
#     num_epochs = 100
#     batch_size = 512
#     learning_rate = 1e-4
#     betas = (0.9, 0.999)
#     weight_decay = 0
#     dropout = 0 
#     d_h = 256
#     n_layer = 2
    
    
#     dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
#     path = os.path.join(dirc + "data/", args.data_filename)

#     dataset = GraphDataset(args.data_filename, path)

#     if args.data_filename == 'graph_stack2':
#         train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.92, 0.04, 0.04])
#     else:
#         train_data, validation_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.1, 0.1])
    
#     # Save test dataset for later use
#     if not os.path.isfile(path + "/test_dataset"):
#         torch.save(test_data, path + "/test_dataset")

#     # d_in = d_out = 2*dataset.Nrf
#     d_in = 2*dataset.M*dataset.Nrf
#     d_out = 2*dataset.Nrf

#     if args.activation not in ['ReLU', 'ELU']:
#             raise KeyError(f'Optimizaer type {args.activation} not supported.\
#                              Use "ReLU" or "ELU" only')
#     elif args.activation == 'ReLU':
#         activation_fn = nn.ReLU()
#     elif args.activation == 'ELU':
#         activation_fn = nn.ELU()


#     rmse_list = []
#     d_h_list = [64, 128, 256, 512]
#     learning_rate = 1e-4
#     # learning_rate_list = [1e-4, 1e-5, 1e-6]
#     batch_size_list = [256, 512, 1024]
#     n_layer_list = [1,2]
#     dropout_list = [0.1, 0.3, 0.5, 0.7]
#     for batch_size in batch_size_list:    
#         for learning_rate in dropout_list:
#             for d_h in d_h_list:
#                 for n_layer in n_layer_list:

#                     model = GraphNeuralNetworkDrop(d_in, d_h, d_out, n_layer, activation_fn=activation_fn, dropout=dropout).to(device)
#                     # model = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=nn.ELU()).to(device)

#                     # Create data loaders.
#                     train_dataloader = GraphDataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
#                     validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

#                     print(model)
#                     # print(f"Hyperparameters: lr = {learning_rate}, wd = {weight_decay}, do = {dropout}")

#                     # loss_fn = nn.MSELoss()
#                     loss_fn = system_model_loss
                    
#                     if args.optimizer not in ['ADAM', 'SGD']:
#                             raise KeyError(f'Optimizaer type {args.optim} not supported.\
#                                             Use "ADAM" or "SGD" only')
#                     elif args.optimizer == 'ADAM':
#                         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
#                     elif args.optimizer == 'SGD':
#                         momentum = 0.9
#                         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

                    

#                     # config = {"epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM"}
#                     config = {"graph": args.data_filename, "epochs": num_epochs, "h_dim": d_h, "mlp_layer": n_layer, "batch_size": batch_size, "lr": learning_rate, 
#                               "weight_decay": weight_decay, "dropout": dropout, "optim": "ADAM", "num_rfchain": dataset.Nrf, 'act': args.activation} # for W&B
#                     if not args.saving_filename: 
#                         saving_filename = f"{config['graph']}_{datetime.now().strftime('%d-%m-%Y-@-%H:%M')}_{config['optim']}_mlp-layer_{n_layer}_lr_{learning_rate} \
#                                             _dh_{d_h}_batch_{batch_size}_act_{config['act']}.pth"
#                     else:
#                         saving_filename = args.saving_filename  

#                     train(train_dataloader, validation_dataloader, model, loss_fn, optimizer, device, num_epochs, config, saving_filename)
                    
#                     # Test phase
#                     test_dataloader = GraphDataLoader(test_data, batch_size=batch_size, shuffle=False)
#                     model_test = GraphNeuralNetwork2(d_in, d_h, d_out, n_layer, activation_fn=activation_fn).to(device)
#                     path = os.path.join(dirc + "models/", saving_filename)
#                     model_test.load_state_dict(torch.load(path)['model_state_dict'])

#                     rmse_list.append(test(test_dataloader, model_test, device))

    
#     print("Done!")
#     print([f"{elem:.3f}" for elem in rmse_list])

   