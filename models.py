import os

os.environ["DGLBACKEND"] = "pytorch"
import torch
from torch import nn
import time
import wandb
import numpy as np
from cmath import inf
from utils import save_model, rmse
# from torch.nn import init
# import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
import dgl.nn.pytorch as dglnn
from datetime import datetime



class GraphNeuralNetwork(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, activation_fn):
        super(GraphNeuralNetwork, self).__init__()
        
        self.init_layer = InitLayer(in_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        self.conv = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConv(in_feats=h_feats, out_feats=h_feats, aggregation_type='mean', activation=activation_fn),
                    'channel2u' : GraphConv(in_feats=h_feats, out_feats=h_feats, aggregation_type='mean', activation=activation_fn)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        # GCN layers
        h = self.conv(g, h)
        h = self.conv(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return out



class InitLayer(nn.Module):

    def __init__(self, in_feats, h_feats, activation=None, embed_in=True):
        super(InitLayer, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.activation = activation
        self.embed_input = embed_in

        self.linear_ue = nn.Linear(in_feats, h_feats) # alternitevly, can be a pre-defined DNN
        self.linear_t = nn.Linear(h_feats, h_feats) # alternitevly, can be a pre-defined DNN


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_ue = graph.num_nodes('user')//graph.batch_size
            num_t = graph.num_nodes('antenna')//graph.batch_size

            if self.embed_input: # Embed pilots to intial node features
                feat_dict = graph.ndata['feat'] # Get users node features --> shape: batch*N_ue X in_feats (2*num_rf)
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.linear_ue(feat_dict['user'])
                
                if self.activation is not None: 
                    feat_dict['user'] = self.activation(feat_dict['user'])

                # Element-wise mean for all features in every batch --> shape: batch X h_feats
                antenna_nodes = feat_dict['user'].view(graph.batch_size, num_ue, -1).mean(dim=1) 
                # antenna_nodes = 0.5 * torch.ones(antenna_nodes.shape).to(torch.cuda.current_device())

                # Embed antenna node feats  --> shape: batch X h_feats
                antenna_nodes = self.linear_t(antenna_nodes)
                if self.activation is not None:
                    antenna_nodes = self.activation(antenna_nodes)
                # print(antenna_nodes[:2])

                # Repeat feature - every antenna node get the same feature yet differs between graphs --> shape: batch*N_t X h_feats
                feat_dict['antenna'] = torch.repeat_interleave(antenna_nodes, num_t, dim=0)
                feat_dict['antenna'] += torch.randn(feat_dict['antenna'].shape).to(torch.cuda.current_device()) # Add noise to make antenna node features distinct

       
            
            else: # Use pilots directly as intialization for node feats
                feat_dict = graph.ndata['feat'] # Get users node features --> shape: batch*N_ue X in_feats
                # Element-wise mean for all features in every batch --> shape: batch X in_feats
                antenna_nodes = feat_dict['user'].view(graph.batch_size, num_ue, -1).mean(dim=1) 
                # Repeat feature - every antenna node get the same feature yet differs between graphs --> shape: batch*N_t X in_feats
                feat_dict['antenna'] = torch.repeat_interleave(antenna_nodes, num_t, dim=0)

            return feat_dict
        


class NormLayer(nn.Module):

    def __init__(self, h_feats, out_feats, embed_in=True):
        super(NormLayer, self).__init__()
        
        self.out_feats = out_feats
        self.h_feats = h_feats
        self.embed_input = embed_in

        self.linear = nn.Linear(h_feats, out_feats) # alternitevly, can be a pre-defined DNN


    def forward(self, graph, feat):
        """
        feat - shape: batch*N_t X h_feat
        """
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_t = graph.num_nodes('antenna')//graph.batch_size
            num_rf = self.out_feats//2

            if self.embed_input: 
                # Embed antenna node feats --> shape: batch X N_t X out_feats (2*num_rf)
                feat = self.linear(feat).view(graph.batch_size, num_t, -1)

                
            feat = torch.complex(feat[:,:,:num_rf], feat[:,:,num_rf:]) # shape: batch X N_t X num_rf (out_feats/2), dtype: complex64
            # feat /= feat.abs()   
            out = feat / feat.abs() # Element-wise normalization to meet unit modulus constraint  

            return out
    



class GraphConv(nn.Module):

    def __init__(self, in_feats, out_feats, aggregation_type, activation=None):
        super(GraphConv, self).__init__()
        
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.lin_comb = nn.Linear(out_feats, out_feats) 

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h") 

        return aggregate_fn   
        

    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():

            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.num_edges()
            #     graph.edata["_edge_weight"] = edge_weight
            #     aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"

            if self.activation is not None:
                # Aggregation stage
                graph.srcdata["h"] = self.activation(self.lin_aggr(feat_src))
                graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"] and update
                                                        # the corresponding dst nodes hidden states at dstdata["h"]
                aggr_message = graph.dstdata["h"]

                # Combination stage
                out = self.lin_comb(self.activation(self.lin_self(feat_dst)) + aggr_message)
                out = self.activation(out)
            
            else:
                # Aggregation stage
                graph.srcdata["h"] = self.lin_aggr(feat_src)
                graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"] and update
                                                        # the corresponding dst nodes hidden states at dstdata["h"]
                aggr_message = graph.dstdata["h"]

                # Combination stage
                out = self.lin_comb(self.lin_self(feat_dst) + aggr_message)
            

            return out



######################

class GraphNeuralNetwork2(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn):
        super(GraphNeuralNetwork2, self).__init__()
        
        self.init_layer = InitLayer(in_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        self.conv = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConv2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn),
                    'channel2u' : GraphConv2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv(g, h)
        # print(h['antenna'][:2])
        h = self.conv(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return out

class MLP(nn.Module):
    def __init__(self, feat_dim, activation, n_layer):
        super(MLP, self).__init__()
        self.linear_list = nn.ModuleList()
        for i in range(n_layer):
            self.linear_list.append(nn.Linear(feat_dim, feat_dim))
        self.activation = activation
        

    def forward(self,x):
        for i in range(len(self.linear_list)):
            x = self.linear_list[i](x)
            x = self.activation(x)

        return x
    

class GraphConv2(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU()):
        super(GraphConv2, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP(feat_dim, activation, n_layer)
        self.mlp_self = MLP(feat_dim, activation, n_layer)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP(feat_dim, activation, n_layer)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h") 

        return aggregate_fn   
        

    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():

            # if edge_weight is not None:
            #     assert edge_weight.shape[0] == graph.num_edges()
            #     graph.edata["_edge_weight"] = edge_weight
            #     aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"] and update
                                                    # the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            out = self.mlp_comb(self.mlp_self(feat_dst) + aggr_message)
            # out = self.activation(out)

            return out





def train(dataloader, validloader, model, loss_fn, optimizer, device, num_epochs, config, filename):
    wandb.init(name= f"gnn1_{datetime.now().strftime('%d-%m-%Y @ %H:%M')}_dh_{config['h_dim']}_mlp-layer_{config['mlp_layer']}_batch_{config['batch_size']}_lr_{config['lr']}", project="gnn_psn_calib", config=config)
    wandb.watch(model)
    best_loss = inf
    # total_step = len(dataloader)
    batch_size = dataloader.batch_size
    model.train()
    start_time = time.time() 
    for epoch in range(num_epochs):
        for i, (g, pilots, combiner, channel, _) in enumerate(dataloader):  
            # Features shape - x: torch.Size([batch_size, 181]), y: torch.Size([batch_size, 7])  
            # Each i is a batch of batch_size smaples
            g, pilots, combiner, channel = g.to(device), pilots.to(device), combiner.to(device), channel.to(device)
            # feat_dict = g.ndata['feat'] # includes only user node features

            # user_feats = g.nodes['user'].data['feat']
            # antenna_feats = g.nodes['antenna'].data['feat']
            # feat_dict = {'antenna': antenna_feats, 'user': user_feats}
            
            # Forward pass
            # pred = model(g, feat_dict)['antenna'] # get antenna elements node features
            # pred = model(g)['antenna']

            
            # Predict the PSN deviation matrices - W
            pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64
            # Process data for loss calculation

            # num_nodes = g.num_nodes('antenna') / batch_size
            # num_rf = pred.shape[2]
            # W = pred.view(batch_size, num_nodes, -1)
            # W = torch.complex(W[:,:,:num_rf], W[:,:,num_rf:])

            lhs = pilots.view(batch_size, -1) # dtype: float32
            rhs = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2), combiner), channel) # dtype: complex64
            rhs = torch.cat((rhs.reshape(batch_size,-1).real, rhs.reshape(batch_size,-1).imag), dim=1) # dtype: float32
            
            loss = loss_fn(lhs, rhs)

            # sanity_check = torch.matmul(torch.mul(w.to(device), combiner), channel)
            # sanity_check = torch.cat((sanity_check.reshape(batch_size,-1).real, sanity_check.reshape(batch_size,-1).imag), dim=1)
            # print(sanity_check)
            # print(lhs)
            # print(torch.isclose(sanity_check, lhs).cpu().numpy())
            # np.savetxt('my_file.txt', torch.isclose(sanity_check, lhs).cpu().numpy().astype(int))
            # print(np.where(torch.isclose(sanity_check, lhs).cpu().numpy() == False))
            # print(sanity_check[2,10])
            # print(lhs[2,10])
            # print(sanity_check[11,13])
            # print(lhs[11,13])
            # exit(1)
            # nn.MSELoss()(lhs.view(batch_size,-1), rhs.view(batch_size,-1))
            # loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.item()

        # validation step
        if (epoch+1) % 2 == 0:
            # valid_loss = validate(validloader, model, loss_fn, device, dataloader.dataset.max, dataloader.dataset.min)
            num_batches = len(validloader)
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for g, pilots, combiner, channel, _ in validloader:
                    # inputs and labels normalization to [0,1]
                    g, pilots, combiner, channel = g.to(device), pilots.to(device), combiner.to(device), channel.to(device)
                    # feat_dict = g.ndata['feat']
                    # user_feats = g.nodes['user'].data['feat']
                    # antenna_feats = g.nodes['antenna'].data['feat']
                    # feat_dict = {'antenna': antenna_feats, 'user': user_feats}
                    
                    # Forward pass
                    pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64

                    # Process data for loss calculation
                    # num_nodes = g.num_nodes('antenna') / batch_size
                    # W = pred.view(batch_size, num_nodes, -1)
                    # W = torch.complex(W[:,:,:num_rf], W[:,:,num_rf:])
                    lhs = pilots.view(batch_size, -1) # dtype: float32
                    rhs = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2), combiner), channel) # dtype: complex64
                    rhs = torch.cat((rhs.reshape(batch_size,-1).real, rhs.reshape(batch_size,-1).imag), dim=1) # dtype: float32

                    valid_loss += loss_fn(lhs, rhs).item()
            
            valid_loss /= num_batches
            model.train()

            print(f"Current validation loss: {valid_loss:.8f}")
            # save model
            if valid_loss < best_loss:
                best_loss = valid_loss
                outfile = '/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/models'
                save_model(epoch+1, model, optimizer, loss_fn, best_loss, config, os.path.join(outfile, filename))
            wandb.log({"loss": loss, "validation loss": valid_loss})
        else:
            wandb.log({"loss": loss})

        
        t = (time.time() - start_time)
        h = int(np.floor(t/3600))
        m = int(np.floor((t-h*3600)/60))
        s = int(np.floor(t-h*3600-m*60))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.8f}, Time: {h:0>2}:{m:0>2}:{s:0>2}")

    wandb.finish()

    # # save last model 
    # outfile = '/ubc/ece/home/ll/grads/idanroth/Desktop/eece_571f_project/data/models'
    # save_model(num_epochs, model, optimizer, loss_fn, best_loss, os.path.join(outfile, os.path.splitext(filename)[0] + '_last.pth'))
    # print(f"Best validation loss: {best_loss:.8f}")



# def validate(dataloader, model, loss_fn, device, max, min): 
#     num_batches = len(dataloader)
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for x, y in dataloader:
#             # inputs and labels normalization to [0,1]
#             x = (x - min)/(max-min)
#             y = (y + PI)/(2 * PI)
#             x, y = x.to(device), y.to(device)
#             pred = model(x)
#             val_loss += loss_fn(pred, y).item()
    
#     val_loss /= num_batches
#     return val_loss


def test(dataloader, model, device):
    size = len(dataloader.dataset) 
    # num_batches = len(dataloader)
    # batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    dev_rmse = 0
    with torch.no_grad():
        for g, _, _, _, psn_dev in dataloader:
            g, psn_dev = g.to(device), psn_dev.to(device)
            # feat_dict = g.ndata['feat']
            
            # Forward pass
            pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), type: Complex
            # num_nodes = g.num_nodes('antenna') / batch_size
            # W = pred.view(batch_size, num_nodes, -1)
            pred = torch.transpose(pred, dim0=1,dim1=2)
            # # Process data for loss calculation
            # num_nodes = g.num_nodes('antenna') / batch_size
            # W = pred.view(batch_size, num_nodes, -1)
            # lhs = pilots
            # rhs = torch.matmul(torch.mul(W.T, combiner), channel)

            # test_loss += loss_fn(lhs.view(batch_size,-1), rhs.view(batch_size,-1)).item()

            dev_rmse += rmse(pred, psn_dev) ############   GET batch of 2 Complex64 matrices  ################
            exit(1)
         
    dev_rmse /= size
 
    print(f"Test Error: \n Avg RMSE: {dev_rmse:.8f} deg. \n")
    
    return dev_rmse