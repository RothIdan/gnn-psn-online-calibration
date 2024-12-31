import os
os.environ["DGLBACKEND"] = "pytorch"
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
import wandb
import numpy as np
from utils import save_model
from dgl import function as fn
from dgl.utils import expand_as_pair
import dgl.nn.pytorch as dglnn


    

class GraphNeuralNetwork(nn.Module):
    """ 
    GNN model for phase shifters network online calibration, which includes an intialization layer, GCN layers and a redout function. 
    It uses recieved pilots as features for the users nodes and the combiner phases as features for the antenna nodes.
    Params:
        in_feats (int): input dimensions dictionary for each initialization MLP with node types as keys.
        h_feats (int): hyperparameter, the hidden dimension of the MLPs.
        out_feats (int): output dimension after final layer.
        conv_layers (int): number of GCN/MPNN layers.
        mlp_layers (int): number of layers in the GCN/MPNN MLPs.
        activation_fn: the Pytorch class activation fiunction.
        dropout (float): dropout probability.
        aggr_fn (str): aggregation function type for GCN/MPNN of antenna update.
        dev (int): the phase deviation "streangth" which correspond to a Uniform distrbution with a range [-dev,dev], for the sigmoid activation function.
    Returns:
        out (float - [batch_size, out_feats]): the PSN phase deviation estimation.
    """
    def __init__(self, in_feats, h_feats, out_feats, conv_layers, mlp_layers, activation_fn, dropout, aggr_fn, dev):
        super(GraphNeuralNetwork, self).__init__()
    
        # Initialization layer
        self.init_layer = InitLayer(in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we update node features and use them
        
        # GCN/MPNN update layers
        self.convs = nn.ModuleList()
        for t in range(conv_layers):
                self.convs.append(dglnn.HeteroGraphConv({
                                    'channel2an' : GraphConv(feat_dim=h_feats, aggregation_type=aggr_fn, n_layer=mlp_layers, activation=activation_fn, dropout=dropout),
                                    'channel2ue' : GraphConv(feat_dim=h_feats, aggregation_type='mean', n_layer=mlp_layers, activation=activation_fn, dropout=dropout)},
                                    aggregate='sum')) # this aggregation is for when more than a single neigbooring node types are connected to the target node
        
        # Final readout function
        self.readout = ReadoutFunction(h_feats, out_feats, dev)


    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        for t in range(len(self.convs)):
            h = self.convs[t](g, h)

        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis)
        out = self.readout(h['antenna'], g.batch_size)

        return out   




class InitLayer(nn.Module):
    """ 
    The initalization layer which intitalize the hidden state/representaiton of the user and antenna nodes.
    Params:
        in_feats (dict): input dimension for the embedding MLP for the 'antenna' and 'user' nodes.
        h_feats (int): hidden layers dimension for the embedding MLP.
        out_feats (int): output dimension for the embedding MLP, the hidden dimension of the nodes hidden representation (hyperparameter).
        activation: the Pytorch class activation fiunction.
    Returns:
        feat_dict (dict): DGL dictionary which holds the graph nodes hidden states. Keys: 'users', 'antenna'.
    """

    def __init__(self, in_feats, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayer, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_dict = nn.ModuleDict()
        
        # Node type-specific NNs
        for ntype in list(in_feats.keys()):
            self.mlp_dict[ntype] = nn.Sequential(nn.Linear(in_feats[ntype], h_feats),
                                                 nn.BatchNorm1d(h_feats),
                                                 activation,
                                                 nn.Linear(h_feats, out_feats),
                                                 nn.BatchNorm1d(out_feats),
                                                 activation)


    def forward(self, graph):
       
        with graph.local_scope(): 

            feat_dict = graph.ndata['feat'] # Get node features dictionary

            if self.embed_input: # Embed pilots to intial node features
                for ntype in graph.ntypes:
                    feat_dict[ntype] = self.mlp_dict[ntype](feat_dict[ntype])

            return feat_dict




class GraphConv(nn.Module):
    """ 
    The spatial-based graph convolution neural network (GCN) with two stages for hidden state update: aggregation and combination.
    Params:
        feat_dim (int): the hidden dimension of the MLPs.
        aggregation_type (str): pooling fuinction used at the aggregation stage. Support: 'mean', 'sum', 'max', 'min', 'lstm', 'concat'.
        n_layer (int): no. of layers for the MLP.
        activation: the Pytorch class activation fiunction.
        dropout (float): dropout probability.
    Returns:
        out: the updated hidden representation of the dpecific node type for each batch of graphs.
    """

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConv, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_self = MLP(feat_dim, feat_dim, activation, n_layer, dropout)

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP(2*feat_dim, feat_dim, activation, n_layer, dropout)

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        # Modify combination NN if 'concat' aggregation is used which concatenates both the mean and the max aggregations
        if aggregation_type == "concat":
            self.mlp_comb = MLP(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
        if aggregator_type not in ['mean', 'sum', 'max', 'min', 'lstm', 'concat']:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\n'
                            'Use "mean", "sum", "max", "min", "lstm", or "concat" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h")
        elif aggregator_type == 'lstm':
            aggregate_fn = self._lstm_reducer 
        elif aggregator_type == 'concat':
            aggregate_fn = self._mean_max_reducer 

        return aggregate_fn   
        

    def _lstm_reducer(self, nodes):
   
        m = nodes.mailbox['m'] # (B, L, D)
        batch_size = m.shape[0]
        h = (m.new_zeros((1, batch_size, self.feat_dim)),
             m.new_zeros((1, batch_size, self.feat_dim)))
        _, (rst, _) = self.lstm(m, h)

        return {'h': rst.squeeze(0)}
    
    
    def _mean_max_reducer(self, nodes):
        """
        Concatenate mean aggregation with max aggregation.

        nodes.mailbox['m'] is of shape: dst_nodes X src_nodes X h_dim
        i.e., contains in dim0 each dst node corresponded src (neighboors) features, 
        dim 1 holds all the src features, and dim2 holds the features themself.
        """
        return {'h': torch.concat((torch.mean(nodes.mailbox['m'], dim=1), torch.max(nodes.mailbox['m'], dim=1).values), dim=1)}



    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():

            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.num_edges()
            #     graph.edata["_edge_weight"] = edge_weight
            #     aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            feat_src, feat_dst = expand_as_pair(feat, graph)

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            out = self.mlp_comb(torch.concat((self.mlp_self(feat_dst), aggr_message), dim=1))

            return out

   


class ReadoutFunction(nn.Module):
    """ 
    Readout function which take the 'antenna' nodes hidden state from the last layer and estimate the PSN phase deviations.
    Params:
        h_feats (int): dimension of the nodes hidden state.
        out_feats (int): output dimension of the estimation target per antenna node.
        dev (int): modify sigmoid activation function range based on the phase deviation "range" prior knowledge. 
    Returns:
        out (float - [batch_size,Nr,out_feats]): phase deviation estimates.
    """

    def __init__(self, h_feats, out_feats, dev):
        super(ReadoutFunction, self).__init__()
        
        self.out_feats = out_feats
        self.h_feats = h_feats

        self.readout = nn.Linear(h_feats, out_feats) # alternitevly, can be a pre-defined DNN
     
        self.sigmoid = nn.Sigmoid()
        self.sigma = np.deg2rad(dev)/(3**0.5) # Uniform dist [-dev,dev] std for the PSN deviation Gaussian distribution


    def forward(self, feat, batch_size):
        """
        feat - shape: batch_size*Nr X h_feats
        """
       
        num_r = feat.shape[0]//batch_size
    
        # Embed antenna node feats --> shape: batch X Nr X out_feats (Nrf or Nrf*Nb)
        feat = self.readout(feat).view(batch_size, num_r, -1) # shape: batch X Nr X out_feats/2 (Nrf or Nrf*Nb)

        # phase estimation
        out = self.sigmoid(feat) * 2*3*self.sigma - 3*self.sigma # 99.7% of the values are whithin +-3sigma

        return out



class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, activation, n_layer, dropout):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList()
        if in_feats != out_feats:
            self.linear_list.append(nn.Sequential(nn.Linear(in_feats, out_feats),
                                                  nn.BatchNorm1d(out_feats),
                                                  activation,
                                                  nn.Dropout(dropout)))
            n_layer -= 1
        
        for i in range(n_layer):
            self.linear_list.append(nn.Sequential(nn.Linear(out_feats, out_feats),
                                                  nn.BatchNorm1d(out_feats),
                                                  activation,
                                                  nn.Dropout(dropout)))
        

    def forward(self,x):
        for i in range(len(self.linear_list)):
            x = self.linear_list[i](x)

        return x
    


################ Training and testing loops ################### 

def train(dataloader, validloader, model, optimizer, loss_fn, device, num_epochs, config, filename, mode):
    wandb.init(name=filename, project="gnn_psn_calib", config=config, mode=mode)
    wandb.watch(model)
    best_loss = np.inf
    batch_size = dataloader.batch_size
    model.train()

    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    start_time = time.time() 
    for epoch in range(num_epochs):
        cumu_loss = 0
        for i, (g, _, _, _, psn_dev) in enumerate(dataloader):  
            # Each i-th iteration gets a batch of batch_size smaples
            # g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
            g, psn_dev = g.to(device), psn_dev.to(device)
            # Nrf, Nr = psn_dev.shape[1], psn_dev.shape[2]
            Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None

            # Predict the PSN deviation matrices - Omega
            pred = model(g) # shape: batch X Nr X Nrf (or Nrf*Nb) , dtype: Float32

            # Process data for loss calculation
            psn_dev = psn_dev.angle().view(batch_size, -1) if Nb is None else torch.transpose(psn_dev.angle(),dim0=3,dim1=2).reshape(batch_size, -1)
            
            w = torch.transpose(pred, dim0=2, dim1=1)
            w = w.reshape(batch_size, -1)

            loss = loss_fn.loss(w, psn_dev)
            cumu_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = cumu_loss / len(dataloader)

        # validation step
        if (epoch+1) % 3 == 0:
            num_batches = len(validloader)
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for g, _, _, _, psn_dev in validloader:
                    g, psn_dev = g.to(device), psn_dev.to(device)

                    Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None

                    pred = model(g) 
                    
                    pred = torch.transpose(pred, dim0=2, dim1=1) 
                    pred = pred.reshape(batch_size, -1)

                    psn_dev = psn_dev.angle().view(batch_size, -1) if Nb is None else torch.transpose(psn_dev.angle(),dim0=3,dim1=2).reshape(batch_size, -1)
    
                    valid_loss += loss_fn.loss(pred, psn_dev).item() 
            
            valid_loss /= num_batches
            model.train()
            scheduler.step(valid_loss)

            print(f"Current validation loss: {valid_loss:.8f}")
            # save model
            if valid_loss < best_loss:
                best_loss = valid_loss
                outfile = "<full_path_name>/models" # NOTE: need to replace with the full path name of the directory which contain the script
                save_model(epoch+1, model, optimizer, best_loss, config, os.path.join(outfile, filename))
            wandb.log({"loss": avg_loss, "validation loss": valid_loss})
        
        else:
            wandb.log({"loss": avg_loss})
        
        
        t = (time.time() - start_time)
        h = int(np.floor(t/3600))
        m = int(np.floor((t-h*3600)/60))
        s = int(np.floor(t-h*3600-m*60))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.8f}, Time: {h:0>2}:{m:0>2}:{s:0>2}")

    wandb.finish()



def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset) 
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    rmse, mse = 0, 0
    with torch.no_grad():
        for g, _, _, _, psn_dev in dataloader:
            g, psn_dev = g.to(device), psn_dev.to(device)
            Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
            Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
            
            # Forward pass
            pred = model(g) 
            pred = torch.transpose(pred, dim0=2, dim1=1) if Nb is None else torch.transpose(pred.reshape(batch_size, Nt, Nrf, Nb), dim0=2, dim1=1)

            pred = pred.reshape(batch_size, -1)
            psn_dev = psn_dev.angle().view(batch_size, -1)

            test_loss += loss_fn.loss(pred, psn_dev).item()
           
            pred = loss_fn.get_modified_pred()

            rmse += torch.mean((pred-psn_dev)**2, dim=1).sum()
            mse += torch.sum((pred-psn_dev)**2, dim=1).sum()
   
    rmse = torch.sqrt(rmse/size)
    test_loss /= num_batches
 
    print(f"Test Error: \n Avg MSE: {mse/size:.8f} rad. per 1 RF chain \n Avg RMSE per parameter: {rmse.rad2deg():.8f} deg. \n Avg loss: {test_loss:.8f} \n")
    
    return rmse
