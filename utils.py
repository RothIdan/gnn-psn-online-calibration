import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
from dgl.data import DGLDataset
from torch.utils.data import Dataset
import torch
from torch import nn
import glob
import pandas as pd
import numpy as np
from functools import partial



# def complex_loss(pilots, combiner, channel, pred):
#     batch_size, Nt, Nrf = pred.shape
#     lhs = torch.complex(pilots[:,:Nrf,:], pilots[:,Nrf:,:])
#     # pilots.view(batch_size, -1) # dtype: float32
#     rhs = torch.matmul(torch.mul(torch.transpose(pred, dim0=1,dim1=2), combiner), channel) # dtype: complex64
#     # Matrix factorization loss
#     return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))


class LossFn:
    """
    Loss function class.
    Params:
        mod (str): 'sys' - system model loss, 
                   'dev' - phase shifter network deviation loss, 
                   'comb' - their combination.
        alpha (float): the magnitude of effect for the 'dev' loss part in mod 'comb'.
    """

    def __init__(self, mod, alpha=1):
        if mod not in ['sys', 'dev', 'comb']:
            raise KeyError(f'Loss type {mod} not supported.\
                             Use "sys", "dev" or "comb" only')
        else:    
            self.mod = mod
        self.alpha = alpha
        self.counter = 0

    def loss(self, pilots, combiner, channel, pred, psn_dev):
        if self.mod == 'dev':
            return self.__deviation_loss(pred, psn_dev)
        else:
            y_pred = torch.matmul(torch.mul(pred, combiner), channel) # dtype: complex64
            if self.mod == 'sys':
                return self.__system_model_loss(pilots, y_pred) 
            else: # mod == 'comb'
                return self.__system_model_loss(pilots, y_pred) + self.alpha * self.__deviation_loss(pred, psn_dev)


    def __system_model_loss(self, lhs, rhs):
        # Can try also matrix factorization loss - torch.mean(torch.norm(lhs - rhs, dim = (1,2))**2)
        return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))
    
    def __deviation_loss(self, pred, psn_dev):
        batch_size = pred.shape[0]
        pred = pred.angle().view(batch_size, -1)
        psn_dev = psn_dev.angle().view(batch_size, -1)
        return torch.mean(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1])
    
    # def __deviation_loss(self, pred, psn_dev):
    #     return torch.mean((torch.norm(pred - psn_dev, dim=(1,2))**2) / (2*pred[0].numel()))
    

def system_model_loss(lhs, rhs):
        # Can try also matrix factorization loss - torch.mean(torch.norm(lhs - rhs, dim = (1,2))**2)
        # return torch.mean((torch.norm(lhs - rhs, dim=(1,2))**2) / (2*lhs[0].numel()))
        return torch.mean((torch.norm(lhs - rhs, dim=1)**2))
    
def deviation_loss(pred, psn_dev):
        batch_size = pred.shape[0]
        pred = pred.angle().view(batch_size, -1)
        psn_dev = psn_dev.angle().view(batch_size, -1)
        return torch.mean(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1])   # Which would be identical as using nn.MSELoss()(pred, psn_dev)


def rmse(pred, psn_dev):
    batch_size = pred.shape[0]
    pred = pred.angle().rad2deg().view(batch_size, -1)
    psn_dev = psn_dev.angle().rad2deg().view(batch_size, -1)
    
    # pred = torch.ones(pred.shape).to(torch.cuda.current_device()) * 0 # check when using mean 
    
    rmse = torch.sqrt(torch.sum((pred-psn_dev)**2, dim=1) / psn_dev.shape[1]) # RMSE calculation of every output of the batch

    return rmse.sum() 



def save_model(epochs, model, optimizer, loss_val, config, outfile):
    """
    Function to save the trained model to disk.
    """
    print(f"Saving final model...")
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss,
                'best_loss': loss_val,
                'batch_size': config['batch_size'],
                'h_dim': config['h_dim'],
                'num_rfchain': config['num_rfchain'],
                'num_meas': config['num_meas'],
                'mlp_layer': config['mlp_layer'],
                'dropout': config['dropout'],
                'num_states': config['num_states']
                }, outfile)

def save_testset(path, data_filename, test_data, dataset):
    graphs = []
    data_dict = {}
    pilots, combiner, channel, psn_dev = [],[],[],[]
    for n in range(len(test_data)):
        graphs.append(test_data[n][0])
        pilots.append(test_data[n][1])
        combiner.append(test_data[n][2])
        channel.append(test_data[n][3])
        psn_dev.append(test_data[n][4])
    
    data_dict['psn_dev'] = np.array(psn_dev)
    data_dict['channel'] = np.array(channel)
    data_dict['pilots'] = np.array(pilots)
    data_dict['combiner'] = np.array(combiner)
    data_dict['num_rfchain'] = dataset.Nrf
    data_dict['indices'] = dataset.indices
    data_dict['num_meas'] = dataset.Q
    data_dict['num_states'] = dataset.Nb
    # data_dict['pilot_len'] = dataset.pilot_len

    dgl.save_graphs(os.path.join(path, data_filename +'_test.dgl'), graphs)
    np.save(os.path.join(path, data_filename +'_test.npy'), data_dict)

    # if 'fixed' in os.path.splitext(data_filename)[0].split('_'):
    #     test_dataset = GraphDatasetFixed(data_filename+'_test', path)
    # else:
    #     test_dataset = GraphDataset(data_filename+'_test', path)

    # torch.save(test_dataset, path + "/test_dataset")
    
    
# class GraphDataset(Dataset):
#     def __init__(self, data_filename, dataset_path="/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/"):
#         super().__init__(name="graph")
#         self.graphs = dgl.load_graphs(os.path.join(dataset_path, data_filename +".dgl"))
#         data_dict = np.load(os.path.join(dataset_path, data_filename +".npy"))
#         self.psn_dev = data_dict['labels'] # PSN deviation matrices
#         self.pilots = data_dict['features']
#         self.channel = data_dict['channel']
#         self.combiner = data_dict['combiner']


#     def __getitem__(self, i):
#         return self.graphs[i], self.pilots[i], self.combiner[i], self.channel[i], self.psn_dev[i]

#     def __len__(self):
#         return len(self.graphs)
    


class GraphDataset(DGLDataset):
    def __init__(self, filename, path="/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/"):
        super().__init__(name=filename, url=None, raw_dir=path)

    def process(self):
        self.graphs, _ = dgl.load_graphs(os.path.join(self.raw_dir, self.name +".dgl"))
        data_dict = np.load(os.path.join(self.raw_dir, self.name +".npy") ,allow_pickle=True).item()

        self.psn_dev = data_dict['psn_dev'] # PSN deviation matrices
        self.pilots = data_dict['pilots']
        self.channel = data_dict['channel']
        self.combiner = data_dict['combiner']
        self.Nrf = data_dict['num_rfchain']
        try:
            self.indices = data_dict['indices']
        except:
            self.indices = None
        try:
            self.Q = data_dict['num_meas']
        except:
            self.Q = 1
        try:
            self.Nb = data_dict['num_states']
        except:
            self.Nb = None
        # self.Nt = data_dict['num_antenna']
        # self.Nue = data_dict['num_user']
        
    def __getitem__(self, i):
        return self.graphs[i], self.pilots[i], self.combiner[i], self.channel[i], self.psn_dev[i]

    def __len__(self):
        return len(self.graphs)
    
    
class GraphDatasetFixed(DGLDataset):
    def __init__(self, filename, path="/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/"):
        super().__init__(name=filename, url=None, raw_dir=path)

    def process(self):
        self.graphs, _ = dgl.load_graphs(os.path.join(self.raw_dir, self.name +".dgl"))
        data_dict = np.load(os.path.join(self.raw_dir, self.name +".npy") ,allow_pickle=True).item()

        self.psn_dev = data_dict['psn_dev'] # PSN deviation matrices
        self.pilots = data_dict['pilots']
        self.channel = data_dict['channel']
        self.combiner = data_dict['combiner']
        self.Nrf = data_dict['num_rfchain']
        try:
            self.indices = data_dict['indices']
        except:
            self.indices = None
        try:
            self.Q = data_dict['num_meas']
        except:
            self.Q = 1
        try:
            self.Nb = data_dict['num_states']
        except:
            self.Nb = None
        # self.Nt = data_dict['num_antenna']
        # self.Nue = data_dict['num_user']
        
    def __getitem__(self, i):
        return self.graphs[i], self.pilots[i], self.combiner, self.channel[i], self.psn_dev[i]

    def __len__(self):
        return len(self.graphs)
    
    
    
    class MLPDataset(Dataset):
        def __init__(self, data_filename, dataset_path='/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/'):
            data_dict = np.load(os.path.join(dataset_path, data_filename +".npy") ,allow_pickle=True).item()

            self.psn_dev = data_dict['psn_dev'] # PSN deviation matrices
            self.pilots = data_dict['pilots']
            self.Nrf = data_dict['num_rfchain']
            try:
                self.M = data_dict['num_meas']
            except:
                self.M = 1
            try:
                self.Nb = data_dict['num_states']
            except:
                self.Nb = None
     
            

        def __len__(self):
            return len(self.pilots)

        def __getitem__(self, i):
            return self.pilots[i], self.psn_dev[i]



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    



class HetroEdgeConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(HetroEdgeConv, self).__init__()
        self.mods = nn.ModuleDict(mods)
        # Do not break if graph has 0-in-degree nodes.
        # Because there is no general rule to add self-loop for heterograph.
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregate

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input edge features - e_feat.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes, with keys as node types.
        """
        
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}

        outputs = {dtype : [] for dtype in g.dsttypes}
        # if isinstance(inputs, tuple) or g.is_block:
        #     if isinstance(inputs, tuple):
        #         src_inputs, dst_inputs = inputs
        #     else:
        #         src_inputs = inputs
        #         dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

        # for canonical_etype in g.canonical_etypes:
        for (stype, etype, dtype) in g.canonical_etypes:
            rel_graph = g[(stype, etype, dtype)]
            # dtype = rel_graph.dsttypes
            # if stype not in src_inputs or dtype not in dst_inputs:
            #     continue
            dstdata = self.mods[f"{(stype, etype, dtype)}"](rel_graph,
                                                            inputs[(stype, etype, dtype)],
                                                            *mod_args.get((stype, etype, dtype), ()),
                                                            **mod_kwargs.get((stype, etype, dtype), {}))
            
            outputs[dtype].append(dstdata)

        # else:
        #     for stype, etype, dtype in g.canonical_etypes:
        #         rel_graph = g[stype, etype, dtype]
        #         if stype not in inputs:
        #             continue
        #         dstdata = self.mods[etype](
        #             rel_graph,
        #             (inputs[stype], inputs[dtype]),
        #             *mod_args.get(etype, ()),
        #             **mod_kwargs.get(etype, {}))
        #         outputs[dtype].append(dstdata)
            
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return torch.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return torch.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return torch.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return torch.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return torch.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = torch.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    else:
        raise KeyError('Invalid cross type aggregator. Must be one of '
                       '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)
   