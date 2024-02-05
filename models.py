import os

os.environ["DGLBACKEND"] = "pytorch"
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import time
import wandb
import numpy as np
from cmath import inf
from utils import save_model, rmse, deviation_loss, system_model_loss
# from torch.nn import init
# import torch.nn.functional as F
from dgl import function as fn
from dgl.utils import expand_as_pair
import dgl.nn.pytorch as dglnn
from datetime import datetime
from utils import HetroEdgeConv



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

    def __init__(self, in_feats, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayer, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        # self.linear_ue = nn.Linear(in_feats, h_feats) # alternitevly, can be a pre-defined DNN
        # self.linear_t = nn.Linear(h_feats, h_feats) # alternitevly, can be a pre-defined DNN
        self.mlp_ue = nn.Sequential(nn.Linear(in_feats, h_feats),
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)
        
        self.mlp_t = nn.Sequential(nn.Linear(out_feats, h_feats),
                                   nn.BatchNorm1d(h_feats),
                                   activation,
                                   nn.Linear(h_feats, out_feats),
                                   nn.BatchNorm1d(out_feats),
                                   activation)


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_ue = graph.num_nodes('user')//graph.batch_size
            num_t = graph.num_nodes('antenna')//graph.batch_size

            feat_dict = graph.ndata['feat'] # Get users node features --> shape: batch*N_ue X in_feats (2*N*N_rf)

            if self.embed_input: # Embed pilots to intial node features
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.mlp_ue(feat_dict['user'])
                
                # if self.activation is not None: 
                #     feat_dict['user'] = self.activation(feat_dict['user'])

                # Element-wise mean for all features in every batch --> shape: batch X h_feats
                # antenna_nodes = feat_dict['user'].view(graph.batch_size, num_ue, -1).mean(dim=1) 
                # antenna_nodes = 0.5 * torch.ones(antenna_nodes.shape).to(torch.cuda.current_device())
                torch.manual_seed(2)
                antenna_nodes = torch.randn(graph.num_nodes('antenna'), self.out_feats).to(torch.cuda.current_device()) #shape: batch*N_t X out_feats

                # Embed antenna node feats  --> shape: batch X h_feats
                # antenna_nodes = self.linear_t(antenna_nodes)
                feat_dict['antenna'] = self.mlp_t(antenna_nodes)
                # feat_dict['antenna'] = antenna_nodes

                # if self.activation is not None:
                #     antenna_nodes = self.activation(antenna_nodes)
        

                # Repeat feature - every antenna node get the same feature yet differs between graphs --> shape: batch*N_t X h_feats
                # feat_dict['antenna'] = torch.repeat_interleave(antenna_nodes, num_t, dim=0)
                # feat_dict['antenna'] += torch.randn(feat_dict['antenna'].shape).to(torch.cuda.current_device()) # Add noise to make antenna node features distinct

       
            # else: # Use pilots directly as intialization for node feats
            #     feat_dict = graph.ndata['feat'] # Get users node features --> shape: batch*N_ue X in_feats
            #     # Element-wise mean for all features in every batch --> shape: batch X in_feats
            #     antenna_nodes = feat_dict['user'].view(graph.batch_size, num_ue, -1).mean(dim=1) 
            #     # Repeat feature - every antenna node get the same feature yet differs between graphs --> shape: batch*N_t X in_feats
            #     feat_dict['antenna'] = torch.repeat_interleave(antenna_nodes, num_t, dim=0)

            return feat_dict
        
    



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



##################################

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

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConv2, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
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


##################################################################################

class GraphNeuralNetworkConcat(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout):
        super(GraphNeuralNetworkConcat, self).__init__()
        
        self.init_layer = InitLayer(in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        # GCN layers
        h = self.conv1(g, h)
        # print(h['antenna'][:2])
        h = self.conv2(g, h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return out



class GraphConvConcatV2(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConvConcatV2, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h_N")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h_N")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h_N")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h_N") 

        return aggregate_fn
    

    def u_cat_v_udf(self, edges):
        return {"e": torch.concat((edges.dst["h"], edges.src["h"]), dim=1)}   
        

    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():
            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            graph.srcdata["h"], graph.dstdata["h"] = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########

            # Aggregation stage
            graph.apply_edges(self.u_cat_v_udf)
            graph.edata["e"] = self.mlp_aggr(graph.edata["e"])

            message_fn = fn.copy_e("e", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"

           
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h_N"]

            # Combination stage
            out = self.mlp_comb(torch.concat((graph.dstdata["h"], aggr_message), dim=1))

            return out


####### GNN version with CSI #########
class GraphEdgeConvOld(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphEdgeConv, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation
        
        # self.mlp_aggr = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        self.mlp_aggr1 = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_aggr2 = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h_N")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h_N")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h_N")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h_N") 

        return aggregate_fn
    

    def u_cat_e_udf(self, edges):
        return {"e": torch.concat((edges.src["h"], edges.data["e"]), dim=1)} # concatenates the src feature "h" with its corresponding edge feature "e" 
                                                                             # and insert the new feature into the edge feature "e"
        

    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():
            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            graph.srcdata["h"], graph.dstdata["h"] = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########

            # Aggregation stage
            # graph.edata["e"] = self.mlp_edge(graph.edata["feat"]) # embed the edge features into a higher dimension
            # graph.apply_edges(self.u_cat_e_udf) # updates each edge feature "e" according to the defined function "u_cat_e_udf"
            # graph.edata["e"] = self.mlp_aggr(graph.edata["e"])

            # #####
            graph.edata["e"] = self.mlp_aggr1(graph.edata["feat"])
            graph.srcdata["h"] = self.mlp_aggr2(graph.srcdata["h"])
            graph.apply_edges(fn.u_add_e("h", "e", "e"))
            # #####

            message_fn = fn.copy_e("e", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"

           
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h_N"]

            # Combination stage
            out = self.mlp_comb(torch.concat((graph.dstdata["h"], aggr_message), dim=1))

            return out

class GraphEdgeConv(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphEdgeConv, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation
        
        # self.mlp_aggr = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_edge = MLP2(2, feat_dim, activation, 2, dropout)
        self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h_N")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h_N")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h_N")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h_N") 

        return aggregate_fn
    

    def u_cat_e_udf(self, edges):
        return {"m": torch.concat((edges.src["h"], edges.data["e"]), dim=1)} # concatenates the src feature "h" with its corresponding edge feature "e" 
                                                                             # and insert the new message feature into the edge feature "m"    

    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():
            # devide to the src nodes hidden states and the dst nodes hidden states (srcdata, dstdata tensors)
            graph.srcdata["h"], graph.dstdata["h"] = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########

            # Aggregation stage
            graph.edata["e"] = self.mlp_edge(graph.edata["e"]) # embed the edge features into a higher dimension
            graph.srcdata["h"] = self.mlp_aggr(graph.srcdata["h"])
            # graph.apply_edges(self.u_cat_e_udf) # updates each edge feature "e" according to the defined function "u_cat_e_udf"
            
            message_fn = fn.u_add_e("h", "e", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"

            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h_N"]

            # Combination stage
            out = self.mlp_comb(torch.concat((self.mlp_self(graph.dstdata["h"]), aggr_message), dim=1))

            return out

class EdgeGraphNeuralNetwork(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout):
        super(EdgeGraphNeuralNetwork, self).__init__()
        self.init_layer = InitLayer2(in_feats, 2*h_feats, h_feats, activation_fn)
        # self.init_layer = InitLayerEdge(in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphEdgeConv(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphEdgeConv(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphEdgeConv(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphEdgeConv(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant

        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv1(g, h)
        # print(h['antenna'][:2])
        h = self.conv2(g, h)
        # h = self.conv3(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return [out]
    
class InitLayerEdge(nn.Module):

    def __init__(self, in_feats, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayerEdge, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_ue = nn.Sequential(nn.Linear(in_feats, h_feats),
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)
        
        self.mlp_t = nn.Sequential(nn.Linear(in_feats, h_feats),
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)
        
        self.mlp_edge = nn.Sequential(nn.Linear(2, h_feats),
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)


    def forward(self, graph):
    
        # with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
        # num_ue = graph.num_nodes('user')//graph.batch_size
        # num_t = graph.num_nodes('antenna')//graph.batch_size

        feat_dict = graph.ndata['feat'] # Get node features --> shape: batch*N_ue X in_feats (2*N*Nrf)

        if self.embed_input: # Embed pilots to intial node features
            # Embed user node feats  --> shape: batch*N_ue X h_feats
            feat_dict['user'] = self.mlp_ue(feat_dict['user'])
            feat_dict['antenna'] = self.mlp_t(feat_dict['antenna'])

            graph[('user', 'channel2a', 'antenna')].edata["feat"] = self.mlp_edge(graph[('user', 'channel2a', 'antenna')].edata["feat"]) # changes graph edge features globally

            # feat_dict['user'] = feat_dict['user'] / (feat_dict['user'].norm(2, dim=1, keepdim=True) + 1e-8)
            # feat_dict['antenna'] = feat_dict['antenna'] / (feat_dict['antenna'].norm(2, dim=1, keepdim=True) + 1e-8)

        return feat_dict





class MLP2(nn.Module):
    def __init__(self, in_feats, out_feats, activation, n_layer, dropout):
        super(MLP2, self).__init__()
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
        # self.activation = activation
        

    def forward(self,x):
        for i in range(len(self.linear_list)):
            x = self.linear_list[i](x)
            # x = self.activation(x)

        return x



##############################################################
class SkipConGnn(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout):
        super(SkipConGnn, self).__init__()
        # self.init_layer = InitLayer2(in_feats, h_feats, h_feats, activation_fn)
        self.init_layer = InitLayer2(in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        # self.init_layer = InitLayer1(in_feats, h_feats, activation_fn)
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv3 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv4 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant                             
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv1(g, h)
        out1 = self.normalization_layer(g, h['antenna'])

        h = self.conv2(g, h)
        out2 = self.normalization_layer(g, h['antenna'])

        h = self.conv3(g, h)
        out3 = self.normalization_layer(g, h['antenna'])

        h = self.conv4(g, h)
        out4 = self.normalization_layer(g, h['antenna'])

        return [out1, out2, out3, out4]
    

class GraphNeuralNetworkConcat2(nn.Module):
    """ 
    The baseline GNN which uses recieved pilots as features for the UE nodes and the combiner phases as features for the antenna nodes.
    in_feats: input dimension for initialization layer
    h_feats: hyperparameter, the hidden dimension of the MLPs
    out_feats: output dimension after normalization layer
    n_layer: number of layers of MLP
    activation_fn: the Pytorch class activation fiunction
    dropout: dropout probability
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout):
        super(GraphNeuralNetworkConcat2, self).__init__()
        # self.init_layer = InitLayer2(in_feats, h_feats, h_feats, activation_fn)
        self.init_layer = InitLayer2(in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        # self.init_layer = InitLayer1(in_feats, h_feats, activation_fn)
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        # self.conv3 = dglnn.HeteroGraphConv({
        #             'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
        #             'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
        #             aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
        #                              # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        # self.conv4 = dglnn.HeteroGraphConv({
        #             'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
        #             'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
        #             aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
        #                              # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv1(g, h)
        # print(h['antenna'][:2])
        h = self.conv2(g, h)
        # h = self.conv3(g, h)
        # h = self.conv4(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return [out]


class InitLayer2(nn.Module):
    """ 
    The initalization layer which intitalize the hidden representaiton of the UE and antenna nodes.
    in_feats: input dimension for the embedding MLP.
    h_feats: hidden layers dimension for the embedding MLP.
    out_feats: output dimension for the embedding MLP, the hidden dimension of the nodes hidden representation (hyperparameter).
    activation: the Pytorch class activation fiunction.
    """

    def __init__(self, in_feats, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayer2, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_ue = nn.Sequential(nn.Linear(in_feats, h_feats), ###### in_feats + 32 (non-tensor with 16 Nt) / 16 (tensor with 8 Nt) for concatenating CSI
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)
        
        self.mlp_t = nn.Sequential(nn.Linear(in_feats, h_feats),
                                   nn.BatchNorm1d(h_feats),
                                   activation,
                                   nn.Linear(h_feats, out_feats),
                                   nn.BatchNorm1d(out_feats),
                                   activation)


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_ue = graph.num_nodes('user')//graph.batch_size
            num_t = graph.num_nodes('antenna')//graph.batch_size

            feat_dict = graph.ndata['feat'] # Get node features --> shape: batch*N_ue X in_feats (2*N*Nrf)

            if self.embed_input: # Embed pilots to intial node features
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.mlp_ue(feat_dict['user'])
                feat_dict['antenna'] = self.mlp_t(feat_dict['antenna'])


                # feat_dict['user'] = feat_dict['user'] / (feat_dict['user'].norm(2, dim=1, keepdim=True) + 1e-8)
                # feat_dict['antenna'] = feat_dict['antenna'] / (feat_dict['antenna'].norm(2, dim=1, keepdim=True) + 1e-8)

            return feat_dict



class GraphNeuralNetworkCSIConcat(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout, Nt):
        super(GraphNeuralNetworkCSIConcat, self).__init__()
        # self.init_layer = InitLayer2(in_feats, h_feats, h_feats, activation_fn)
        self.init_layer = InitLayerCSI(in_feats+2*Nt, in_feats, 2*h_feats, h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        # self.init_layer = InitLayer1(in_feats, h_feats, activation_fn)
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        # self.conv3 = dglnn.HeteroGraphConv({
        #             'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
        #             'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
        #             aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
        #                              # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        # self.conv4 = dglnn.HeteroGraphConv({
        #             'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
        #             'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
        #             aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
        #                              # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv1(g, h)
        # print(h['antenna'][:2])
        h = self.conv2(g, h)
        # h = self.conv3(g, h)
        # h = self.conv4(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return [out]

class InitLayerCSI(nn.Module):

    def __init__(self, in_feats_ue, in_feats_t, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayerCSI, self).__init__()
        
        self.in_feats = in_feats_t
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_ue = nn.Sequential(nn.Linear(in_feats_ue, h_feats), ###### in_feats + 32 (non-tensor with 16 Nt) / 16 (tensor with 8 Nt) for concatenating CSI
                                    nn.BatchNorm1d(h_feats),
                                    activation,
                                    nn.Linear(h_feats, out_feats),
                                    nn.BatchNorm1d(out_feats),
                                    activation)
        
        self.mlp_t = nn.Sequential(nn.Linear(in_feats_t, h_feats),
                                   nn.BatchNorm1d(h_feats),
                                   activation,
                                   nn.Linear(h_feats, out_feats),
                                   nn.BatchNorm1d(out_feats),
                                   activation)


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_ue = graph.num_nodes('user')//graph.batch_size
            num_t = graph.num_nodes('antenna')//graph.batch_size

            feat_dict = graph.ndata['feat'] # Get node features --> shape: batch*N_ue X in_feats (2*N*Nrf)
            # Concatenate CSI to UE recieved pilot
            feat_dict['user'] = torch.concat((feat_dict['user'], graph.edges['channel2a'].data['e'].reshape(graph.batch_size*num_ue, -1)), dim=1)

            if self.embed_input: # Embed pilots to intial node features
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.mlp_ue(feat_dict['user'])
                feat_dict['antenna'] = self.mlp_t(feat_dict['antenna'])


                # feat_dict['user'] = feat_dict['user'] / (feat_dict['user'].norm(2, dim=1, keepdim=True) + 1e-8)
                # feat_dict['antenna'] = feat_dict['antenna'] / (feat_dict['antenna'].norm(2, dim=1, keepdim=True) + 1e-8)

            return feat_dict
        

class GraphConvConcat(nn.Module):
    """ 
    The graph convolution neural network with two stages: aggregation and combination.
    feat_dim: the hidden dimension of the MLPs.
    aggregation_type: pooling fuinction used at the aggregation stage: 'mean', 'sum', 'max', 'min', 'lstm', 'concat'.
    n_layer: no. of layers for the MLP.
    activation: the Pytorch class activation fiunction.
    dropout: dropout probability.
    """

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConvConcat, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            # recieved_signal = graph.ndata["user"]
            out = self.mlp_comb(torch.concat((self.mlp_self(feat_dst), aggr_message), dim=1))
            # out = self.activation(out)
            # out = out / (out.norm(2, dim=1, keepdim=True) + 1e-8)

            return out

class GraphConvUE(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConvUE, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim+2*32+2*16, feat_dim, activation, n_layer, dropout)
        self.mlp_self = MLP2(2*32, feat_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            recieved_signal = graph.nodes['user'].data['feat']
            # out = self.mlp_comb(torch.concat((self.mlp_self(torch.concatenate((feat_dst, recieved_signal), dim=1)), aggr_message), dim=1))
            out = self.mlp_comb(torch.concat((self.mlp_self(recieved_signal), aggr_message), dim=1)) # y2
            # out = self.mlp_comb(torch.concat((feat_dst, self.mlp_self(recieved_signal), aggr_message), dim=1)) # y3
            # out = self.activation(out)
            # out = out / (out.norm(2, dim=1, keepdim=True) + 1e-8)

            return out
    

class GraphConvMP(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConvMP, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim+2*32+2*16, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(2*32+2*16, feat_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim + 2*32, feat_dim, activation, n_layer, dropout)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            input_feat = graph.dstdata["feat"]
            out = self.mlp_comb(torch.concat((feat_dst, input_feat, aggr_message), dim=1))
  
            # out = self.activation(out)
            # out = out / (out.norm(2, dim=1, keepdim=True) + 1e-8)

            return out
        

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
                # Embed antenna node feats --> shape: batch X N_t X out_feats (2*Nrf or 2*Nrf*Nb)
                feat = self.linear(feat).view(graph.batch_size, num_t, -1)

                
            feat = torch.complex(feat[:,:,:num_rf], feat[:,:,num_rf:]) # shape: batch X N_t X out_feats/2 (Nrf or Nrf*Nb), dtype: complex64

            feat[:,0,0] = 1 # Force it to eliminate ambiguity

            out = feat / feat.abs() # Element-wise normalization to meet unit modulus constraint  

            return out



###################################
class GraphNeuralNetworkConcat3(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, in_feats, h_feats, out_feats, n_layer, activation_fn, dropout):
        super(GraphNeuralNetworkConcat3, self).__init__()
        self.init_layer = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum')
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                    'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='concat', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        # self.conv3 = dglnn.HeteroGraphConv({
        #             'channel2a' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
        #             'channel2u' : GraphConvConcat(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)},
        #             aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
        #                              # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.normalization_layer = NormLayer(h_feats, out_feats)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        
        # GCN layers
        h = self.conv1(g, h)
        # print(h['antenna'][:2])
        h = self.conv2(g, h)
        # h = self.conv3(g, h)
        # h = F.relu(h)
        # Processing antenna embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, h['antenna'])
        return out



class GraphConvConcat3(nn.Module):

    def __init__(self, in_dim, out_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphConvConcat3, self).__init__()
        
        self.in_dim = in_dim
        self.h_dim = out_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(in_dim, out_dim, activation, n_layer, dropout)
        self.mlp_self = MLP2(out_dim, out_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*out_dim, out_dim, activation, n_layer, dropout)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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
            feat_src, feat_dst = expand_as_pair(feat, graph) ##### make sure that this really give the updated features of srcdata and dstdata ##########
            # maybe can just use: feat_src, feat_dst = graph.srcdata["h"], graph.dstdata["h"] if updating them at the end?????

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            # recieved_signal = graph.ndata["user"]
            out = self.mlp_comb(torch.concat((self.mlp_self(feat_dst), aggr_message), dim=1))
            # out = self.activation(out)
            # out = out / (out.norm(2, dim=1, keepdim=True) + 1e-8)

            return out



class InitLayer1(nn.Module):

    def __init__(self, in_feats, h_feats, activation=None, embed_in=True):
        super(InitLayer1, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_ue = nn.Sequential(nn.Linear(in_feats, h_feats),
                                    nn.BatchNorm1d(h_feats),
                                    activation)
        
        self.mlp_t = nn.Sequential(nn.Linear(in_feats, h_feats),
                                   nn.BatchNorm1d(h_feats),
                                   activation)


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            num_ue = graph.num_nodes('user')//graph.batch_size
            num_t = graph.num_nodes('antenna')//graph.batch_size

            feat_dict = graph.ndata['feat'] # Get node features --> shape: batch*N_ue X in_feats (2*N*Nrf)

            if self.embed_input: # Embed pilots to intial node features
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.mlp_ue(feat_dict['user'])
                feat_dict['antenna'] = self.mlp_t(feat_dict['antenna'])


            return feat_dict


class MLPBenchmark(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, activation, n_hid_layer, dropout):
        super(MLPBenchmark, self).__init__()
        self.linear_list = nn.ModuleList()
        if in_feats != h_feats:
            self.linear_list.append(nn.Sequential(nn.Linear(in_feats, h_feats),
                                                  nn.BatchNorm1d(h_feats),
                                                  activation,
                                                  nn.Dropout(dropout)))
            n_hid_layer -= 1
        for i in range(n_hid_layer):
            self.linear_list.append(nn.Sequential(nn.Linear(h_feats, h_feats),
                                                  nn.BatchNorm1d(h_feats),
                                                  activation,
                                                  nn.Dropout(dropout)))
            
        self.linear_list.append(nn.Sequential(nn.Linear(h_feats, out_feats),
                                                  nn.BatchNorm1d(out_feats),
                                                  activation,
                                                  nn.Dropout(dropout)))
        

    def forward(self,x):
        for i in range(len(self.linear_list)):
            x = self.linear_list[i](x)
            # x = self.activation(x)

        return x

class EdgeConv(nn.Module):
    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(EdgeConv, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_aggr = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN
        # self.lin_self = nn.Linear(in_feats, out_feats) # alternitevly, can be a pre-defined DNN

        # An optional MLP mapping which operates over the mapping of the previous hidden state 
        # of the current updated node and the aggregated information added together
        self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)
        # self.lin_comb = nn.Linear(out_feats, out_feats) 
    
    def __init__(self, input_dim, node_dim, **kwargs):
        super(EdgeConv, self).__init__()
        self.lin = MLP([input_dim, 32])
        self.res_lin = Lin(node_dim, 32)
        self.bn = BN(32)
        #self.reset_parameters()

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

    def concat_message_function(self, edges):
        return {'out': torch.cat([edges.src['h'], edges.data['feat']], axis=1)}
    # return {'out': torch.cat([edges.src['hid'], edges.dst['hid'], edges.data['feat']], axis=1)}
    
    def forward(self, graph, feat, edge_weight=None):
       
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            message_fn = fn.copy_edge('out', 'm')
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h"
            
            # Aggregation stage
            graph.srcdata["h"] = feat_src
            # graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn)
         
            # g.srcdata['hid'] = feat_src
            # g.dstdata['hid'] = feat_dst
            
            graph.apply_edges(self.concat_message_function)

            g.edata['out'] = self.lin(g.edata['out'])
            
            graph.update_all(message_fn, aggregate_fn)
            return graph.dstdata['h'] + self.res_lin(feat_dst)




###### GNN with prediciton ######

class GraphNeuralNetworkConcatDecision2(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, h_feats, out_feats, n_layer, activation_fn, dropout, M, Nrf):
        super(GraphNeuralNetworkConcatDecision2, self).__init__()
        self.init_layer = InitLayerDecision(h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them
        self.conv1 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvAntenna2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf),
                    'channel2u' : GraphConvUser2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
        self.conv2 = dglnn.HeteroGraphConv({
                    'channel2a' : GraphConvAntenna2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf),
                    'channel2u' : GraphConvUser2(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)},
                    aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                     # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant

        self.pred_layer1 = PredLayer(h_feats, out_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)
        self.pred_layer2 = PredLayer(h_feats, out_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        # GCN layers
        h = self.conv1(g, h)
        out1 = self.pred_layer1(g, h)
        h = self.conv2(g, h)
        out2 = self.pred_layer2(g, h)

        return [out1, out2]
    
class GraphConvUser2(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0, M=16, Nrf=2):
        super(GraphConvUser2, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_comb = MLP2(feat_dim + 2*M*Nrf, feat_dim, activation, n_layer, dropout)
    

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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

        # with graph.local_scope():

        feat_src, feat_dst = expand_as_pair(feat, graph) # previous user hidden state
        # feat_src = sub_graph.nodes['antenna'].data['feat'] # current antenna hidden state ####### double check!!!!!!!!! ########
        # feat_src, feat_dst = feat['antenna'], feat['user'] # current antenna hidden state, previous user hidden state

        message_fn = fn.copy_u("h", "m")
        aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

        
        # Aggregation stage
        graph.srcdata["h"] = self.mlp_aggr(feat_src)
        graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                    # and update the corresponding dst nodes hidden states at dstdata["h"]
        aggr_message = graph.dstdata["h"]

        # Combination stage
        pilots = graph.nodes['user'].data['pilot']
        out = self.mlp_comb(torch.concat((aggr_message, pilots), dim=1)) # dim: 2*h_dim + 2*N*Nrf --> h_dim
        graph.nodes['user'].data['feat'] = out # update node hidden state for following prediction

        return out

class GraphConvAntenna2(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0, M=16, Nrf=2):
        super(GraphConvAntenna2, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_comb = MLP2(2*Nrf*(M+1)+feat_dim, feat_dim, activation, n_layer, dropout)
    

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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

            # _, feat_dst = expand_as_pair(feat, graph) # src for the users, and dst for antennas
            feat_src = graph.nodes['user'].data['feat']
            # feat_dst = feat['antenna']

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            graph.srcdata["h"] = self.mlp_aggr(feat_src)
            graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                        # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = graph.dstdata["h"]

            # Combination stage
            Nue = graph.num_nodes('user')//graph.batch_size
            Nt = graph.num_nodes('antenna')//graph.batch_size
            pilots_mean = graph.nodes['user'].data['pilot'].view(graph.batch_size, Nue, -1).mean(dim=1)
            prev_pred = graph.nodes['antenna'].data['pred']

            # Update antenna hidden state / generates antenna new message
            out = self.mlp_comb(torch.concat((prev_pred, aggr_message, pilots_mean.repeat_interleave(Nt, dim=0)), dim=1)) # dim: 2*Nrf*(N+1) + h_dim  --> h_dim
            # graph.nodes['antenna'].data['feat'] = out # update node hidden state for following user hidden state update

            return out   


class GraphNeuralNetworkConcatDecision(nn.Module):
    """ 
    n_layer: number of layers of MLP
    """
    def __init__(self, h_feats, out_feats, n_layer, activation_fn, dropout, M, Nrf):
        super(GraphNeuralNetworkConcatDecision, self).__init__()
        self.init_layer = InitLayerDecision(h_feats, activation_fn) ### should not be with g.local scope if we updte node features and use them

        self.conv1_antenna = GraphConvAntenna(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)
        self.conv1_user = GraphConvUser(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)
        self.conv2_antenna = GraphConvAntenna(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)
        self.conv2_user = GraphConvUser(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout, M=M, Nrf=Nrf)

        self.pred_layer1 = PredLayer(h_feats, out_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)
        self.pred_layer2 = PredLayer(h_feats, out_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout)

    
    def forward(self, g):
        # Initial feats embedding for users and antenna nodes to get the feat dict - h 
        h = self.init_layer(g)
        # GCN layers
        h['antenna'] = self.conv1_antenna(g, h)
        h['user'] = self.conv1_user(g, h)
        out1 = self.pred_layer1(g, h)

        h['antenna'] = self.conv2_antenna(g, h)
        h['user'] = self.conv2_user(g, h)
        out2 = self.pred_layer2(g, h)

        return [out1, out2]

class InitLayerDecision(nn.Module):

    def __init__(self, feat_dim, activation=None, embed_in=False):
        super(InitLayerDecision, self).__init__()
        
        # self.in_feats = in_feats
        self.feat_dim = feat_dim
        # self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in

        self.mlp_ue = nn.Sequential(nn.Linear(feat_dim, 2*feat_dim), ###### in_feats + 32
                                    nn.BatchNorm1d(2*feat_dim),
                                    activation,
                                    nn.Linear(2*feat_dim, feat_dim),
                                    nn.BatchNorm1d(feat_dim),
                                    activation)


    def forward(self, graph):
       
        with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
            Nue = graph.num_nodes('user')//graph.batch_size
            Nt = graph.num_nodes('antenna')//graph.batch_size

            # Initialize nodes hidden state / messages
            # torch.manual_seed(2)
            feat_dict = {}
            feat_dict['user'] = torch.randn((graph.batch_size*Nue, self.feat_dim)).to(torch.cuda.current_device())
            feat_dict['antenna'] = torch.randn((graph.batch_size*Nt, self.feat_dim)).to(torch.cuda.current_device())

            if self.embed_input: # Embed pilots to intial node features
                # Embed user node feats  --> shape: batch*N_ue X h_feats
                feat_dict['user'] = self.mlp_ue(feat_dict['user'])
                # feat_dict['antenna'] = self.mlp_t(feat_dict['antenna'])

            return feat_dict

class GraphConvAntenna(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0, M=16, Nrf=2):
        super(GraphConvAntenna, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_comb = MLP2(2*Nrf*(M+1)+feat_dim, feat_dim, nn.Tanh(), n_layer, dropout)
    

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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
       
        sub_graph = graph[('user', 'channel2a', 'antenna')]
        with sub_graph.local_scope():

            # feat_src, feat_dst = expand_as_pair(feat, sub_graph) # src for the users, and dst for antennas
            feat_src, feat_dst = feat['user'], feat['antenna']

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            sub_graph.srcdata["h"] = self.mlp_aggr(feat_src)
            sub_graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                        # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = sub_graph.dstdata["h"]

            # Combination stage
            Nue = graph.num_nodes('user')//graph.batch_size
            Nt = graph.num_nodes('antenna')//graph.batch_size
            pilots_mean = graph.nodes['user'].data['pilot'].view(graph.batch_size, Nue, -1).mean(dim=1)
            prev_pred = graph.nodes['antenna'].data['pred']

            # Update antenna hidden state / generates antenna new message
            out = self.mlp_comb(torch.concat((prev_pred, aggr_message, pilots_mean.repeat_interleave(Nt, dim=0)), dim=1)) # dim: 2*Nrf*(N+1) + h_dim  --> h_dim
            # graph.nodes['antenna'].data['feat'] = out # update node hidden state for following user hidden state update

            return out

class GraphConvUser(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0, M=16, Nrf=2):
        super(GraphConvUser, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        self.mlp_comb = MLP2(2*feat_dim + 2*M*Nrf, feat_dim, nn.Tanh(), n_layer, dropout)
    

        # Optional LSTM aggregation
        if aggregation_type == "lstm":
            self.lstm = nn.LSTM(feat_dim, feat_dim, batch_first=True)
        
        if aggregation_type == "concat":
            self.mlp_comb = MLP2(3*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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

        sub_graph = graph[('antenna', 'channel2u', 'user')]
        with sub_graph.local_scope():

            # _, feat_dst = expand_as_pair(feat, sub_graph) # previous user hidden state
            # feat_src = sub_graph.nodes['antenna'].data['feat'] # current antenna hidden state ####### double check!!!!!!!!! ########
            feat_src, feat_dst = feat['antenna'], feat['user'] # current antenna hidden state, previous user hidden state

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            
            # Aggregation stage
            sub_graph.srcdata["h"] = self.mlp_aggr(feat_src)
            sub_graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = sub_graph.dstdata["h"]

            # Combination stage
            pilots = graph.nodes['user'].data['pilot']
            out = self.mlp_comb(torch.concat((feat_dst, aggr_message, pilots), dim=1)) # dim: 2*h_dim + 2*N*Nrf --> h_dim
            # graph.nodes['user'].data['feat'] = out # update node hidden state for following prediction

            return out
    
class PredLayer(nn.Module):

    def __init__(self, h_feats, out_feats, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(PredLayer, self).__init__()
        
        self.out_feats = out_feats
        self.h_feats = h_feats
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)

        self.mlp_aggr = MLP2(h_feats, h_feats, activation, n_layer, dropout)
        self.linear = MLP2(2*h_feats, out_feats, nn.Sigmoid(), n_layer, dropout)
        # self.linear = nn.Linear(2*h_feats, out_feats) # alternitevly, can be a pre-defined DNN


    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min", "lstm", "concat"]:
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
        """LSTM reducer
        NOTE(zihao): lstm reducer with default schedule (degree bucketing)
        is slow, we could accelerate this with degree padding in the future.
        """
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

    
    def forward(self, graph, feat):
        """
        feat - shape: batch*N_t X h_feat
        """
       
        # with graph.local_scope(): # should we do it, or should we update g.ndata['feat] = feat_dict????
        Nt = graph.num_nodes('antenna')//graph.batch_size
        Nrf = self.out_feats//2
        
        # Aggregating only the updated user messages
        sub_graph = graph[('user', 'channel2a', 'antenna')]
        with sub_graph.local_scope():
            feat_src, feat_dst = feat['user'], feat['antenna']
            # feat_src, feat_dst = expand_as_pair(feat, sub_graph) # src for the users, and dst for antennas

            message_fn = fn.copy_u("h", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors fetures on all corresponding edges "m" into dst "h"

            sub_graph.srcdata["h"] = self.mlp_aggr(feat_src)
            sub_graph.update_all(message_fn, aggregate_fn) # The following update takes srcdata["h"], compute the aggregate message on the correspoindg edge "m",
                                                        # and update the corresponding dst nodes hidden states at dstdata["h"]
            aggr_message = sub_graph.dstdata["h"]

            # Embed antenna node feats --> shape: batch X N_t X out_feats (2*Nrf or 2*Nrf*Nb)
            feat = self.linear(torch.concat((feat_dst, aggr_message), dim=1)).view(graph.batch_size, Nt, -1)

                
            feat = torch.complex(feat[:,:,:Nrf], feat[:,:,Nrf:]) # shape: batch X N_t X out_feats/2 (Nrf or Nrf*Nb), dtype: complex64

            feat[:,0,0] = 1 # Force it to eliminate ambiguity

            out = feat / feat.abs() # Element-wise normalization to meet unit modulus constraint  

            up_pred = out.reshape(graph.batch_size*Nt, -1)
            # Updates the prediciton in the graph features for future use
            graph.nodes['antenna'].data['pred'] = torch.concat((up_pred.real, up_pred.imag), dim=1) # check if graph is really updated!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            return out
    


################## 3D Edge GNN ##################

class Edge3DGNN(nn.Module):
    """ 
    The baseline GNN which uses recieved pilots as features for the UE nodes and the combiner phases as features for the antenna nodes.
    in_feats: input dimension dictionary for initialization layer.
    h_feats: the hidden dimension of the MLPs.
    out_feats: output dimension after normalization layer.
    conv_layers: number of GCN layers.
    mlp_layers: number of hidden layers in the MLPs.
    activation_fn: the Pytorch class activation fiunction.
    dropout: dropout probability.
    """
    def __init__(self, in_feats, h_feats, out_feats, conv_layers, mlp_layers, activation_fn, dropout):
        super(Edge3DGNN, self).__init__()

        self.init_layer = InitLayer3D(in_feats, 2*h_feats, h_feats, activation_fn)
    
        # self.conv1 = GraphEdgeConvLayer(h_feats, mlp_layer, activation_fn, dropout)
        # self.conv2 = GraphEdgeConvLayer(h_feats, mlp_layer, activation_fn, dropout)
        self.convs = nn.ModuleList()
        for t in range(conv_layers):
            self.convs.append(GraphEdgeConvLayer(h_feats, mlp_layers, activation_fn, dropout))
       
        self.normalization_layer = NormLayer3D(h_feats, out_feats)

    
    def forward(self, g):
        # Initial hidden representation embeddings for all edge types to get the edge features dictionary with keys as the canonical edge types
        e_feats = self.init_layer(g)
        
        # Edge GCN layers
        for t in range(len(self.convs)):
            e_feats = self.convs[t](g, e_feats)
        # e_feat = self.conv1(g, e_feat)
        # e_feat = self.conv2(g, e_feat)
       
        # Processing PSN edges embeddings to get PSN deviations (further node-level anaylsis + deal with constraints)
        out = self.normalization_layer(g, e_feats[('rfchain', 'psn', 'antenna')])

        return [out]


class GraphEdgeConvLayer(nn.Module):
    def __init__(self, h_feats, n_layer, activation_fn, dropout):
        super(GraphEdgeConvLayer, self).__init__()
        self.aggregation = HetroEdgeConv({"('antenna', 'channel', 'user')" : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                                          "('user', 'channel', 'antenna')" : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                                          "('rfchain', 'psn', 'antenna')"  : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                                          "('antenna', 'psn', 'rfchain')"  : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),          
                                          "('user', 'pilot', 'rfchain')"   : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout),
                                          "('rfchain', 'pilot', 'user')"   : GraphEdgeAggregation(feat_dim=h_feats, aggregation_type='mean', n_layer=n_layer, activation=activation_fn, dropout=dropout) },
                                          aggregate='sum') # aggregates the "final" hidden states of the dst node which connects to several src node types into a real final hidden state 
                                                           # have other buit-in options such as max etc. SINCE we use bipartite graph, it is not reallt relevant
                                      

        self.combination = GraphEdgeCombination(feat_dim=h_feats, n_layer=n_layer, activation=activation_fn, dropout=dropout)
    
    
    def forward(self, graph, e_feats):
       # Aggregation 
       # yields a temporary nodes features dictionary with keys as dtype - n_feats, used for updating the edge features in the combination step 
       n_feats = self.aggregation(graph, e_feats)

       # Combination
       # Updates the edge features
       out = self.combination(graph, n_feats, e_feats) # out holds the updated edge features dictionary, updated version of e_feats

       return out



class GraphEdgeAggregation(nn.Module):

    def __init__(self, feat_dim, aggregation_type, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphEdgeAggregation, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.aggrgate_fn = self.get_aggregate_fn(aggregation_type)
        self.activation = activation

        self.mlp_dicts = nn.ModuleDict({"('antenna', 'channel', 'user')" : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                        "('user', 'channel', 'antenna')" : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                        "('rfchain', 'psn', 'antenna')"  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                        "('antenna', 'psn', 'rfchain')"  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                        "('user', 'pilot', 'rfchain')"   : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                        "('rfchain', 'pilot', 'user')"   : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)})
        # self.mlp_aggr = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_self = MLP2(feat_dim, feat_dim, activation, n_layer, dropout)
        # self.mlp_comb = MLP2(2*feat_dim, feat_dim, activation, n_layer, dropout)

        
    def get_aggregate_fn(self, aggregator_type):
    # aggregator type: mean, sum, max, min
        if aggregator_type not in ['mean', 'sum', 'max', "min"]:
            raise KeyError(f'Aggregator type {aggregator_type} not supported.\
                             Use "mean", "sum", "max", or "min" only')
        elif aggregator_type == 'mean':
            aggregate_fn = fn.mean(msg="m", out="h_N")
        elif aggregator_type == 'sum':
            aggregate_fn = fn.sum(msg="m", out="h_N")
        elif aggregator_type == 'max':
            aggregate_fn = fn.max(msg="m", out="h_N")
        elif aggregator_type == 'min':
            aggregate_fn = fn.min(msg="m", out="h_N") 

        return aggregate_fn
    

    def u_cat_e_udf(self, edges):
        return {"m": torch.concat((edges.src["h"], edges.data["e"]), dim=1)} # concatenates the src feature "h" with its corresponding edge feature "e" 
                                                                             # and insert the new message feature into the edge feature "m"    

    def forward(self, graph, feat):
        '''
        graph: DGL sub graph of a relation (dtype, etype, stype).
        feat: Conatins the edge feature from the dictionary e_feats['canonical_etype'].
        '''

        with graph.local_scope():

            # graph.edata["e"] = feat
            graph.edata["e"] = self.mlp_dicts[f"{graph.canonical_etypes[0]}"](feat)
            message_fn = fn.copy_e("e", "m")
            aggregate_fn = self.aggrgate_fn # aggregates src neghibors "m" into dst "h_N"
            
            graph.update_all(message_fn, aggregate_fn) # The following update takes edata["e"], compute the aggregate message on the correspoindg edge "m",
                                                       # and update the corresponding dst nodes hidden states at dstdata["h_N"]
            aggr_message = graph.dstdata["h_N"] # shape: batch_size * num_nodes(dtype) X h_feat

            return aggr_message


class GraphEdgeCombination(nn.Module):
    def __init__(self, feat_dim, n_layer, activation=nn.ReLU(), dropout=0):
        super(GraphEdgeCombination, self).__init__()
        
        self.feat_dim = feat_dim
        self.n_layer = n_layer
        self.activation = activation

        # self.src_mlp_dict = nn.ModuleDict({'user' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                    'antenna'     : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                    'rfchian'   : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)})
        
        # self.dst_mlp_dict = nn.ModuleDict({'user' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                    'antenna'     : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                    'rfchain'   : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)})
        
        # self.edge_mlp_dict = nn.ModuleDict({'channel' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                     'psn'     : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                     'pilot'   : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)})

        # self.mlp_dicts = nn.ModuleDict({'channel' : nn.ModuleDict({'src'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'dst'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'edge' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}),

        #                                 'psn'     : nn.ModuleDict({'src'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'dst'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'edge' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}),

        #                                 'pilot'   : nn.ModuleDict({'src'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'dst'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
        #                                                            'edge' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}) })
        
        self.mlp_dicts = nn.ModuleDict({'channel' : nn.ModuleDict({'antenna'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'user'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'channel' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}),

                                        'psn'     : nn.ModuleDict({'rfchain'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'antenna'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'psn' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}),

                                        'pilot'   : nn.ModuleDict({'user'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'rfchain'  : MLP2(feat_dim, feat_dim, activation, n_layer, dropout),
                                                                   'pilot' : MLP2(feat_dim, feat_dim, activation, n_layer, dropout)}) })

           
    
    def u_add_e_add_v_udf(self, edges):
        return {"e_N": edges.src["h"] + edges.data["e"] + edges.dst["h"]} # sum the src feature "h" with the dst feature "h" and their corresponding edge feature "e" 
                                                                          # and insert the new message feature into the edge feature "e_N" 

    def forward(self, graph, n_feats, e_feats):

        # outputs = {}
        # for (stype, etype, dtype) in graph.canonical_etypes:

        #     sub_graph = graph[(stype, etype, dtype)]
        #     with sub_graph.local_scope():
        #         sub_graph.srcdata["h"] = self.mlp_dicts[etype][stype](n_feats[stype])
        #         sub_graph.dstdata["h"] = self.mlp_dicts[etype][dtype](n_feats[dtype])
        #         sub_graph.edata["e"] = self.mlp_dicts[etype][etype](e_feats[(stype, etype, dtype)])

        #         sub_graph.apply_edges(self.u_add_e_add_v_udf) # Sums the hidden representations of feat_src, feat_dst, and feat_edge for updating the edge feature
                
        #         outputs[(stype, etype, dtype)] = sub_graph.edata["e_N"]

        Nue = graph.num_nodes('user')//graph.batch_size
        Nt = graph.num_nodes('antenna')//graph.batch_size
        Nrf = graph.num_nodes('rfchain')//graph.batch_size
        outputs = {}
        rel_list = [('user', 'channel', 'antenna'), ('user', 'pilot', 'rfchain'), ('rfchain', 'psn', 'antenna')]
        shape = {'user' : Nue, 'antenna' : Nt, 'rfchain' : Nrf}

        for (stype, etype, dtype) in rel_list:
            sub_graph = graph[(stype, etype, dtype)]
            with sub_graph.local_scope():
                sub_graph.srcdata["h"] = self.mlp_dicts[etype][stype](n_feats[stype])
                sub_graph.dstdata["h"] = self.mlp_dicts[etype][dtype](n_feats[dtype])
                sub_graph.edata["e"] = self.mlp_dicts[etype][etype](e_feats[(stype, etype, dtype)])

                sub_graph.apply_edges(self.u_add_e_add_v_udf) # Sums the hidden representations of feat_src, feat_dst, and feat_edge for updating the edge feature
                outputs[(stype, etype, dtype)] = sub_graph.edata["e_N"]

                # Inserting the opposite edges hidden representation based on the previous calculation.
                # Edges on both direction hold the same hidden representations  
                reverse_feat = sub_graph.edata["e_N"].reshape(graph.batch_size, shape[stype]*shape[dtype], -1) 
                if self.feat_dim%2 == 0: # hidden dimension is even
                    dim = self.feat_dim//2
                    reverse_feat = torch.transpose(torch.complex(reverse_feat[:,:,:dim], reverse_feat[:,:,dim:]), dim0=2, dim1=1)
                else: # hidden dimension is odd, need to work seperately on the last element
                    dim = self.feat_dim//2 + 1
                    temp = torch.complex(reverse_feat[:,:,:self.feat_dim//2], reverse_feat[:,:,self.feat_dim//2:-1])
                    reverse_feat = torch.transpose(torch.concat((temp, reverse_feat[:,:,-1:]), dim=2), dim0=2, dim1=1)

                reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, shape[stype], shape[dtype]), dim0=3, dim1=2)
                reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, shape[stype]*shape[dtype]), dim0=2, dim1=1)
                if self.feat_dim%2 == 0: # hidden dimension is even
                    reverse_feat = torch.concat([reverse_feat.real, reverse_feat.imag], dim=2)
                else: 
                    reverse_feat = torch.concat([reverse_feat.real[:,:,:-1], reverse_feat.imag[:,:,:-1], reverse_feat.real[:,:,-1:]], dim=2)
    
                outputs[(dtype, etype, stype)] = reverse_feat.reshape(-1, self.feat_dim)


        # (stype, etype, dtype) = ('user', 'channel', 'antenna')
        # sub_graph = graph[(stype, etype, dtype)]
        # with sub_graph.local_scope():
        #     sub_graph.srcdata["h"] = self.mlp_dicts[etype][stype](n_feats[stype])
        #     sub_graph.dstdata["h"] = self.mlp_dicts[etype][dtype](n_feats[dtype])
        #     sub_graph.edata["e"] = self.mlp_dicts[etype][etype](e_feats[(stype, etype, dtype)])

        #     sub_graph.apply_edges(self.u_add_e_add_v_udf) # Sums the hidden representations of feat_src, feat_dst, and feat_edge for updating the edge feature
        #     outputs[(stype, etype, dtype)] = sub_graph.edata["e_N"]

        #     # Inserting the opposite edges hidden representation based on the previous calculation.
        #     # Edges on both direction hold the same hidden representations  
        #     reverse_feat = sub_graph.edata["e_N"].reshape(graph.batch_size, Nue*Nt, -1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         dim = self.feat_dim//2
        #         reverse_feat = torch.transpose(torch.complex(reverse_feat[:,:,:dim], reverse_feat[:,:,dim:]), dim0=2, dim1=1)
        #     else: # hidden dimension is odd, need to work seperately on the last element
        #         dim = self.feat_dim//2 + 1
        #         temp = torch.complex(reverse_feat[:,:,:self.feat_dim//2], reverse_feat[:,:,self.feat_dim//2:-1])
        #         reverse_feat = torch.transpose(torch.concat((temp, reverse_feat[:,:,-1:]), dim=2), dim0=2, dim1=1)

        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nue, Nt), dim0=3, dim1=2)
        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nt*Nue), dim0=2, dim1=1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         reverse_feat = torch.concat([reverse_feat.real, reverse_feat.imag], dim=2)
        #     else: 
        #         reverse_feat = torch.concat([reverse_feat.real[:,:,:-1], reverse_feat.imag[:,:,:-1], reverse_feat.real[:,:,-1:]], dim=2)
 
        #     outputs[(dtype, etype, stype)] = reverse_feat.reshape(-1, self.feat_dim)


        # (stype, etype, dtype) = ('user', 'pilot', 'rfchain')
        # sub_graph = graph[(stype, etype, dtype)]
        # with sub_graph.local_scope():
        #     sub_graph.srcdata["h"] = self.mlp_dicts[etype][stype](n_feats[stype])
        #     sub_graph.dstdata["h"] = self.mlp_dicts[etype][dtype](n_feats[dtype])
        #     sub_graph.edata["e"] = self.mlp_dicts[etype][etype](e_feats[(stype, etype, dtype)])

        #     sub_graph.apply_edges(self.u_add_e_add_v_udf) # Sums the hidden representations of feat_src, feat_dst, and feat_edge for updating the edge feature
        #     outputs[(stype, etype, dtype)] = sub_graph.edata["e_N"]

        #     reverse_feat = sub_graph.edata["e_N"].reshape(graph.batch_size, Nue*Nrf, -1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         dim = self.feat_dim//2
        #         reverse_feat = torch.transpose(torch.complex(reverse_feat[:,:,:dim], reverse_feat[:,:,dim:]), dim0=2, dim1=1)
        #     else: # hidden dimension is odd, need to work seperately on the last element
        #         dim = self.feat_dim//2 + 1
        #         temp = torch.complex(reverse_feat[:,:,:self.feat_dim//2], reverse_feat[:,:,self.feat_dim//2:-1])
        #         reverse_feat = torch.transpose(torch.concat((temp, reverse_feat[:,:,-1:]), dim=2), dim0=2, dim1=1)

        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nue, Nrf), dim0=3, dim1=2)
        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nrf*Nue), dim0=2, dim1=1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         reverse_feat = torch.concat([reverse_feat.real, reverse_feat.imag], dim=2)
        #     else: 
        #         reverse_feat = torch.concat([reverse_feat.real[:,:,:-1], reverse_feat.imag[:,:,:-1], reverse_feat.real[:,:,-1:]], dim=2)
 
        #     outputs[(dtype, etype, stype)] = reverse_feat.reshape(-1, self.feat_dim)
        
        
        # (stype, etype, dtype) = ('rfchain', 'psn', 'antenna')
        # sub_graph = graph[(stype, etype, dtype)]
        # with sub_graph.local_scope():
        #     sub_graph.srcdata["h"] = self.mlp_dicts[etype][stype](n_feats[stype])
        #     sub_graph.dstdata["h"] = self.mlp_dicts[etype][dtype](n_feats[dtype])
        #     sub_graph.edata["e"] = self.mlp_dicts[etype][etype](e_feats[(stype, etype, dtype)])

        #     sub_graph.apply_edges(self.u_add_e_add_v_udf) # Sums the hidden representations of feat_src, feat_dst, and feat_edge for updating the edge feature
        #     outputs[(stype, etype, dtype)] = sub_graph.edata["e_N"]

        #     reverse_feat = sub_graph.edata["e_N"].reshape(graph.batch_size, Nt*Nrf, -1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         dim = self.feat_dim//2
        #         reverse_feat = torch.transpose(torch.complex(reverse_feat[:,:,:dim], reverse_feat[:,:,dim:]), dim0=2, dim1=1)
        #     else: # hidden dimension is odd, need to work seperately on the last element
        #         dim = self.feat_dim//2 + 1
        #         temp = torch.complex(reverse_feat[:,:,:self.feat_dim//2], reverse_feat[:,:,self.feat_dim//2:-1])
        #         reverse_feat = torch.transpose(torch.concat((temp, reverse_feat[:,:,-1:]), dim=2), dim0=2, dim1=1)

        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nrf, Nt), dim0=3, dim1=2)
        #     reverse_feat = torch.transpose(reverse_feat.reshape(graph.batch_size, dim, Nrf*Nt), dim0=2, dim1=1)
        #     if self.feat_dim%2 == 0: # hidden dimension is even
        #         reverse_feat = torch.concat([reverse_feat.real, reverse_feat.imag], dim=2)
        #     else: 
        #         reverse_feat = torch.concat([reverse_feat.real[:,:,:-1], reverse_feat.imag[:,:,:-1], reverse_feat.real[:,:,-1:]], dim=2)
 
        #     outputs[(dtype, etype, stype)] = reverse_feat.reshape(-1, self.feat_dim)



        return outputs




class InitLayer3D(nn.Module):
    """ 
    The initalization layer which intitalize the hidden representaiton of the edges.
    in_feats: input dimensions dictionary for each embedding MLP with canonical edge types as keys, e.g., in_feats[('rfchain', 'psn2an', 'antenna')].
    h_feats: hidden layers dimension for the embedding MLPs.
    out_feats: output dimension for the embedding MLP, the hidden dimension of the nodes hidden representation (hyperparameter).
    activation: the Pytorch class activation fiunction.
    """

    def __init__(self, in_feats, h_feats, out_feats, activation=None, embed_in=True):
        super(InitLayer3D, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.out_feats = out_feats
        self.activation = activation
        self.embed_input = embed_in
        
        self.mlp_dict = nn.ModuleDict()
        
        # for canonical_etype in list(in_feats.keys()):
        #     self.mlp_dict[canonical_etype] = nn.Sequential(nn.Linear(in_feats[canonical_etype], h_feats),
        #                                                    nn.BatchNorm1d(h_feats),
        #                                                    activation,
        #                                                    nn.Linear(h_feats, out_feats),
        #                                                    nn.BatchNorm1d(out_feats),
        #                                                    activation)
        for etype in list(in_feats.keys()):
            self.mlp_dict[etype] = nn.Sequential(nn.Linear(in_feats[etype], h_feats),
                                                           nn.BatchNorm1d(h_feats),
                                                           activation,
                                                           nn.Linear(h_feats, out_feats),
                                                           nn.BatchNorm1d(out_feats),
                                                           activation)


    def forward(self, graph):
       
        with graph.local_scope(): 
            # Nue = graph.num_nodes('user')//graph.batch_size
            # Nt = graph.num_nodes('antenna')//graph.batch_size
            # Nrf = graph.num_nodes('rfchain')//graph.batch_size

            feat_dict = graph.edata['e'] # Get edge features dict with keys as the edge canonical types (stype, etype, dtype)
                                         # shapes: batch*num_edges['etype'] X in_feats['etype'] 

            if self.embed_input: # Embed features to intial edges hidden representations
                # Shapes: batch*num_edges['etype'] X h_feats 
                # for canonical_etype in graph.canonical_etypes:
                for (stype, etype, dtype) in graph.canonical_etypes:
                    feat_dict[(stype, etype, dtype)] = self.mlp_dict[etype](feat_dict[(stype, etype, dtype)])

            return feat_dict
        

class NormLayer3D(nn.Module):

    def __init__(self, h_feats, out_feats):
        super(NormLayer3D, self).__init__()
        
        self.out_feats = out_feats
        # self.h_feats = h_feats

        self.linear = nn.Linear(h_feats, out_feats) # alternitevly, can be a pre-defined DNN


    def forward(self, graph, feat):
        """
        feat - shape: batch*Nt*Nrf X h_feat 
        """
       
        with graph.local_scope(): 
            Nt = graph.num_nodes('antenna')//graph.batch_size
            Nrf = graph.num_nodes('rfchain')//graph.batch_size
            

            feat = self.linear(feat).view(graph.batch_size, Nt*Nrf, -1) # shape: batch X Nt*Nrf X out_feats (= 2 or 2*Nb)   
            feat = torch.complex(feat[:,:,:self.out_feats//2], feat[:,:,self.out_feats//2:]) # shape: batch X Nt*Nrf X out_feats/2 (= 1 or Nb), dtype: complex64

            # Constraints
            feat[:,0,0] = 1 # Force it to eliminate ambiguity
            out = feat / feat.abs() # Element-wise normalization to meet unit modulus constraint  

            return out



################ Training and testing loops ################### 

def train(dataloader, validloader, model, optimizer, device, num_epochs, config, filename, mode):
    wandb.init(name=filename, project="gnn_psn_calib", config=config, mode=mode)
    wandb.watch(model)
    best_loss = inf
    batch_size = dataloader.batch_size
    model.train()

    # scheduler = StepLR(optimizer, step_size=150, gamma=0.2) # every step_size epochs: lr <-- lr * gamma
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=num_epochs)

    start_time = time.time() 
    for epoch in range(num_epochs):
        cumu_loss = 0
        for i, (g, pilots, combiner, channel, psn_dev) in enumerate(dataloader):  
            # Each i is a batch of batch_size smaples
            g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)

            # Predict the PSN deviation matrices - W
            pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64

            # Process data for loss calculation

            # lhs = pilots.view(batch_size, -1) # dtype: float32
            # rhs = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2), combiner), channel) # dtype: complex64
            # rhs = torch.cat((rhs.reshape(batch_size,-1).real, rhs.reshape(batch_size,-1).imag), dim=1) # dtype: float32
            
            # loss = loss_fn(lhs, rhs)
            loss = 0
            for w in pred:
                w = torch.transpose(w, dim0=1, dim1=2)
                loss += deviation_loss(w, psn_dev)
            # loss = deviation_loss(torch.transpose(pred, dim0=1, dim1=2), psn_dev)
            cumu_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = cumu_loss / len(dataloader)
        # scheduler.step()

        # validation step
        if (epoch+1) % 3 == 0:
            # valid_loss = validate(validloader, model, loss_fn, device, dataloader.dataset.max, dataloader.dataset.min)
            num_batches = len(validloader)
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for g, pilots, combiner, channel, psn_dev in validloader:
                    # inputs and labels normalization to [0,1]
                    g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                    
                    pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64

                    valid_loss += deviation_loss(torch.transpose(pred[-1], dim0=1, dim1=2), psn_dev).item()
            
            valid_loss /= num_batches
            model.train()

            print(f"Current validation loss: {valid_loss:.8f}")
            # save model
            if valid_loss < best_loss:
                best_loss = valid_loss
                outfile = '/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/models'
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

    # # save last model 
    # outfile = '/ubc/ece/home/ll/grads/idanroth/Desktop/eece_571f_project/data/models'
    # save_model(num_epochs, model, optimizer, loss_fn, best_loss, os.path.join(outfile, os.path.splitext(filename)[0] + '_last.pth'))
    # print(f"Best validation loss: {best_loss:.8f}")

def tensor_train(dataloader, validloader, model, optimizer, device, num_epochs, config, filename, mode):
    wandb.init(name=filename, project="gnn_psn_calib", config=config, mode=mode)
    wandb.watch(model)
    best_loss = inf
    batch_size = dataloader.batch_size
    model.train()
    start_time = time.time() 
    for epoch in range(num_epochs):
        cumu_loss = 0
        for i, (g, pilots, combiner, channel, psn_dev) in enumerate(dataloader):  
            # Each i is a batch of batch_size smaples
            g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
            _, Nrf, Nt, Nb = psn_dev.shape
            # Predict the PSN deviation matrices - W
            pred = model(g) # shape: batch X N_t X N_rf * N_b (out_feats/2), dtype: Complex64
            # pred = torch.transpose(pred.reshape(batch_size, Nt, Nrf, Nb), dim0=1, dim1=2) # shape: batch X N_rf X N_t X N_b

            # loss = deviation_loss(pred, psn_dev)

            loss = 0
            for w in pred:
                w = torch.transpose(w.reshape(batch_size, Nt, Nrf, Nb), dim0=1, dim1=2) # shape: batch X N_rf X N_t X N_b
                loss += deviation_loss(w, psn_dev)
           

            cumu_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = cumu_loss / len(dataloader)

        # validation step
        if (epoch+1) % 3 == 0:
            # valid_loss = validate(validloader, model, loss_fn, device, dataloader.dataset.max, dataloader.dataset.min)
            num_batches = len(validloader)
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for g, pilots, combiner, channel, psn_dev in validloader:
                    # inputs and labels normalization to [0,1]
                    g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                    _, Nrf, Nt, Nb = psn_dev.shape
                    pred = model(g) # shape: batch X N_t X N_rf * Nb (out_feats/2), dtype: Complex64
                    pred = torch.transpose(pred[-1].reshape(validloader.batch_size, Nt, Nrf, Nb), dim0=1, dim1=2)

                    valid_loss += deviation_loss(pred, psn_dev).item()
            
            valid_loss /= num_batches
            model.train()

            print(f"Current validation loss: {valid_loss:.8f}")
            # save model
            if valid_loss < best_loss:
                best_loss = valid_loss
                outfile = '/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/models'
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


def edge_train(dataloader, validloader, model, optimizer, device, num_epochs, config, filename, mode):
    wandb.init(name=filename, project="gnn_psn_calib", config=config, mode=mode)
    wandb.watch(model)
    best_loss = inf
    batch_size = dataloader.batch_size
    model.train()

    # scheduler = StepLR(optimizer, step_size=150, gamma=0.2) # every step_size epochs: lr <-- lr * gamma
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(dataloader), epochs=num_epochs)

    start_time = time.time() 
    for epoch in range(num_epochs):
        cumu_loss = 0
        for i, (g, pilots, combiner, channel, psn_dev) in enumerate(dataloader):  
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
                w = w.reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb)
                loss += deviation_loss(w, psn_dev)
           
            cumu_loss += loss.item()

            # loss = 0
            # for w in pred:
            #     w = w.reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64
            #     lhs = torch.concat((pilots.reshape(batch_size, -1).real, pilots.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
            #     rhs = torch.matmul(torch.mul(torch.kron(torch.ones(Q,1).to(device), w), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
            #     rhs = torch.concat((rhs.reshape(batch_size, -1).real, rhs.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

            #     loss += system_model_loss(lhs, rhs)
           
            # cumu_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = cumu_loss / len(dataloader)
        # scheduler.step()

        # validation step
        if (epoch+1) % 3 == 0:
            # valid_loss = validate(validloader, model, loss_fn, device, dataloader.dataset.max, dataloader.dataset.min)
            num_batches = len(validloader)
            size = len(validloader.dataset) 
            model.eval()
            valid_loss = 0
            dev_rmse = 0
            with torch.no_grad():
                for g, pilots, combiner, channel, psn_dev in validloader:
            
                    g, pilots, combiner, channel, psn_dev = g.to(device), pilots.to(device), combiner.to(device), channel.to(device), psn_dev.to(device)
                    
                    pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), dtype: Complex64

                    w = pred[-1].reshape(batch_size, Nrf, Nt) if Nb is None else w.reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64
                    # lhs = torch.concat((pilots.reshape(batch_size, -1).real, pilots.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32
                    # rhs = torch.matmul(torch.mul(torch.kron(torch.ones(Q,1).to(device), w), combiner), channel) # Shape: batch X Q*Nrf X Nue, dtype: complex64
                    # rhs = torch.concat((rhs.reshape(batch_size, -1).real, rhs.reshape(batch_size, -1).imag), dim=1) # Shape: batch X 2*Q*Nrf*Nue, dtype: float32

                    # valid_loss += system_model_loss(lhs, rhs).item()
                    valid_loss += deviation_loss(w, psn_dev).item()
                    dev_rmse += rmse(w, psn_dev) ############   GET batch of 2 Complex64 matrices  ################

            
            valid_loss /= num_batches
            dev_rmse /= size
            model.train()

            print(f"Current validation loss: {valid_loss:.8f}")
            print(f"Current validation Avg RMSE: {dev_rmse:.8f} deg.")

            # save model
            if valid_loss < best_loss:
                best_loss = valid_loss
                outfile = '/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/models'
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
    num_batches = len(dataloader)
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
            pred = torch.transpose(pred[-1], dim0=1,dim1=2)
            # # Process data for loss calculation
            # num_nodes = g.num_nodes('antenna') / batch_size
            # W = pred.view(batch_size, num_nodes, -1)
            # lhs = pilots
            # rhs = torch.matmul(torch.mul(W.T, combiner), channel)

            test_loss += deviation_loss(pred, psn_dev).item()

            dev_rmse += rmse(pred, psn_dev) ############   GET batch of 2 Complex64 matrices  ################
         
    dev_rmse /= size
    test_loss /= num_batches
 
    print(f"Test Error: \n Avg RMSE: {dev_rmse:.8f} deg. \n Avg loss: {test_loss:.8f} \n")
    
    return dev_rmse


def tensor_test(dataloader, model, device):
    size = len(dataloader.dataset) 
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    dev_rmse = 0
    with torch.no_grad():
        for g, _, _, _, psn_dev in dataloader:
            g, psn_dev = g.to(device), psn_dev.to(device)
            # feat_dict = g.ndata['feat']
            _, Nrf, Nt, Nb = psn_dev.shape
            # Forward pass
            pred = model(g) # shape: batch X N_t X N_rf (out_feats/2), type: Complex
           
            pred = torch.transpose(pred[-1].reshape(pred[-1].shape[0], Nt, Nrf, Nb), dim0=1, dim1=2)
      

            test_loss += deviation_loss(pred, psn_dev).item()

            dev_rmse += rmse(pred, psn_dev) ############   GET batch of 2 Complex64 matrices  ################
         
    dev_rmse /= size
    test_loss /= num_batches
 
    print(f"Test Error: \n Avg RMSE: {dev_rmse:.8f} deg. \n Avg loss: {test_loss:.8f} \n")
    
    return dev_rmse


def edge_test(dataloader, model, device):
    size = len(dataloader.dataset) 
    num_batches = len(dataloader)
    batch_size = dataloader.batch_size
    model.eval()
    test_loss = 0
    dev_rmse = 0
    with torch.no_grad():
        for g, _, _, _, psn_dev in dataloader:
            g, psn_dev = g.to(device), psn_dev.to(device)
            
            Nrf, Nt = psn_dev.shape[1], psn_dev.shape[2]
            Nb = psn_dev.shape[3] if psn_dev.dim() == 4 else None
            
            # Forward pass
            pred = model(g) # shape: batch X Nt*Nrf X 1 or Nb, dtype: Complex64
            pred = pred[-1].reshape(batch_size, Nrf, Nt) if Nb is None else pred[-1].reshape(batch_size, Nrf, Nt, Nb) # dtype: Complex64

            test_loss += deviation_loss(pred, psn_dev).item()
            dev_rmse += rmse(pred, psn_dev) ############   GET batch of 2 Complex64 matrices  ################
         
    dev_rmse /= size
    test_loss /= num_batches
 
    print(f"Test Error: \n Avg RMSE: {dev_rmse:.8f} deg. \n Avg loss: {test_loss:.8f} \n")
    
    return dev_rmse