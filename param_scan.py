import torch
from torch import nn
# from torch.utils.data import DataLoader # Iterable which helps to treat data

import numpy as np
import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.dataloading import GraphDataLoader
import argparse

from utils import GraphDataset, system_model_loss, GraphDatasetFixed, deviation_loss
from models import GraphNeuralNetworkConcat, GraphNeuralNetworkConcat2

import wandb, yaml, pprint, functools
from multiprocessing import Process



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--data_filename', type=str, help=" file name of data for both .dgl and .npy, e.g., 'graph1'", required=True)
    parser.add_argument('-s', '--saving_filename', type=str, help=" file name of the model parameters")
    # parser.add_argument('-id', '--id', help=" sweep id from wandb", required=True)

    return parser.parse_args()


def train(train_data, validation_data, d_in, d_out, M, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        # model = GraphNeuralNetworkDrop(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        # model = GraphNeuralNetworkConcat(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        model = GraphNeuralNetworkConcat2(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)

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


def tensor_train(train_data, validation_data, d_in, d_out, M, config=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, this config will be set by Sweep Controller
        config = wandb.config

        train_dataloader = GraphDataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
        validation_dataloader = GraphDataLoader(dataset=validation_data, batch_size=config.batch_size, shuffle=False, drop_last=True, num_workers=4)

        # model = GraphNeuralNetworkDrop(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        # model = GraphNeuralNetworkConcat(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)
        model = GraphNeuralNetworkConcat2(d_in, config.d_h, d_out, config.n_layer, activation_fn=nn.ReLU(), dropout=config.dropout).to(device)

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
                pred = torch.transpose(pred.reshape(config.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2)
                # y_pred = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2).repeat(1,M,1), combiner), channel) # dtype: complex64


                # loss = system_model_loss(pilots, y_pred) + alpha * deviation_loss(pred, psn_dev)
                # loss = system_model_loss(pilots, y_pred)
                loss = deviation_loss(pred, psn_dev)
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
                        pred = torch.transpose(pred.reshape(config.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2)
                        # valid_loss += system_model_loss(pilots, y_pred).item()
                        valid_loss += deviation_loss(pred, psn_dev).item()

                valid_loss /= len(validation_dataloader)

                model.train()
            
                wandb.log({"loss": loss, "validation loss": valid_loss})

            else:
                wandb.log({"loss": loss})
            
            if (epoch+1) == 100 and valid_loss > 0.03: # Early stopping
                    break





if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    torch.manual_seed(2)
    np.random.seed(2)

    args = parse_args() # getting all the init args from input

    with open('sweep_config.yaml', 'r') as stream:
        sweep_config = yaml.safe_load(stream)
    
    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="gnn_psn_calib") 
    
    dirc = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/"
    path = os.path.join(dirc + "data/", args.data_filename)

    if 'fixed' in os.path.splitext(args.data_filename)[0].split('_'):
        dataset = GraphDatasetFixed(args.data_filename, path)
    else:
        dataset = GraphDataset(args.data_filename, path)

    train_data, validation_data = dgl.data.utils.split_dataset(dataset, [0.88, 0.12], random_state=2)
    
    d_in = 2*dataset.M*dataset.Nrf
    if 'tensor' in os.path.splitext(args.data_filename)[0].split('_'):
        d_out = 2*dataset.Nrf*dataset.Nb
    else:
        d_out = 2*dataset.Nrf
    
    # device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device2 = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # train_func1 = functools.partial(train, device1, train_data, validation_data, d_in, d_out, dataset.M)
    # train_func2 = functools.partial(train, device2, train_data, validation_data, d_in, d_out, dataset.M)

    # wandb.agent(sweep_id, function=train_func1, count=50)
    # wandb.agent(sweep_id, function=train_func2, count=50)

    train_func = functools.partial(train, train_data, validation_data, d_in, d_out, dataset.M)

    wandb.agent(sweep_id, function=train_func, count=50)
    # train(train_data, validation_data, d_in, d_out, dataset.M)
   
   
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

   