import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import glob
import numpy as np
import pandas as pd
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--filename', type=str, help=".dgl file name of data, e.g., 'graph1'", required=True)
    parser.add_argument('-n', '--num_graphs', type=int, help="dataset size", default=128000)

    return parser.parse_args()



def generate_pilots(Nrf, Nt, Nue, dl, snr_db=None):
    f = 28e9 # transmission frequency
    wvlen = 3e8/f # wave length
    d = wvlen/2 # elements spacing


    P = np.deg2rad(np.random.randint(-180,180,size=(Nrf,Nt)), dtype=np.float32) 
    P = np.exp(1j * P) # Analog combiner is unit modulus

    W = np.deg2rad(np.random.randint(-10,10,size=(Nrf,Nt)), dtype=np.float32) # PSN deviations - estimation goal
    W = np.exp(1j * W) # dtype = complex64, shape: Nrf X Nt

    # Channel realizations
    # DoA
    a = np.array([np.arange(Nt)]) * (((2*d*np.pi)/wvlen)) # Constants of the steering vector
    doa = np.zeros((Nue, dl))
    doa[:,] = np.random.randint(-90,90,size=(Nue,1)) # Duplicate DoA to all columns 
    spread = np.random.randint(-10,10,size=(Nue,dl-1))
    doa[:,1:] += spread # Adding the spread to the DoA to all columns except the first one to represent the multipath
    A = np.exp(-1j * (np.transpose(a) * np.sin(np.deg2rad(doa.reshape(-1)))), dtype=np.complex64)
    # Gain - complex gaussian
    G = np.zeros((Nue*dl, Nue), dtype=np.complex64)
    for n in range(Nue):
        re = np.random.normal(0, np.sqrt(1/2), dl)
        im = np.random.normal(0, np.sqrt(1/2), dl)
        g = re + 1j*im
        G[n*dl:n*dl+dl, n] = g

    H = np.matmul(A, G) # dtype = complex64, shape: Nt X Nue

    # Generating pilots according to system model
    Y = np.matmul(np.multiply(W, P), H) # dtype = complex64, shape: Nrf X Nue

    # Additive noise
    if snr_db is not None:
        sigma2_z = Y.var() / (10**(snr_db/10))
        re_z = np.random.normal(0, np.sqrt(sigma2_z/2), (Nrf,Nue))
        im_z = np.random.normal(0, np.sqrt(sigma2_z/2), (Nrf,Nue))
        Z = re_z + 1j*im_z
        Y = Y + Z

    Y = np.vstack([Y.real, Y.imag], dtype=np.float32)

    return  W, Y, H, P



if __name__ == "__main__":
    start_time = time.time()
    args = parse_args() # getting all the inti args from input
    path = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/" + args.filename
    graph_data = {}

    for fname in glob.glob(path + "/*.csv"):
        df = pd.read_csv(fname)

        src = torch.tensor(df['src'].to_numpy())
        dst = torch.tensor(df['dst'].to_numpy())
        # get nodes names and relations from fname
        canonical_etype = tuple(os.path.basename(os.path.splitext(fname)[0]).split('-'))
        # build a relation dictionary 
        graph_data[canonical_etype] = (src, dst)

    g = dgl.heterograph(graph_data)
    print(f"Graph feature types:\n nodes: {g.ntypes}, edges: {g.canonical_etypes}")
    # print(g.ntypes)
    # print(g.etypes)
    # print(g.canonical_etypes)
    # print(g.num_nodes())
    # print(g.num_edges())

    # graphs, train_graphs, valid_graphs, test_graphs = [],[],[],[]
    # train_dict, valid_dict, test_dict = {},{},{}

    graphs = []
    data_dict = {}

    # System model parameters
    num_t = g.num_nodes('antenna') # no. of antenna elements
    num_ue = g.num_nodes('user') # no. of UEs
    num_rf = num_t//8 # no. of RF chains
    dl = 3 # no. of channel multipath
    snr_db = None #0 # For no AWGN use the value: None

    print(f"System model parameters:\n no. antenna: {num_t}, no. users: {num_ue}, no. RF chains: {num_rf}, no. multipath: {dl}")

    # train_dict['rfchain'] = num_rf
    # valid_dict['rfchain'] = num_rf
    # test_dict['rfchain'] = num_rf
    data_dict['rfchain'] = num_rf
    data_dict['antenna'] = num_t
    data_dict['user'] = num_ue
    psn_dev, pilots, channel, combiner = [],[],[],[]

    for n in range(args.num_graphs):

        W, Y, H, P = generate_pilots(num_rf, num_t, num_ue, dl, snr_db)
        psn_dev.append(W)
        pilots.append(Y)
        channel.append(H)
        combiner.append(P)

        # Create a graph and add it with its features to the list of graphs
        g = dgl.heterograph(graph_data)
        # create graph features
        # node_features = {}

        
        # Add node feats only for user nodes! 
        # Antenna node feats would be embedded using the user feats inside the gnn model
        g.nodes['user'].data['feat'] = torch.transpose(torch.tensor(Y), dim0=0,dim1=1) # Y: shape 2N_rf X N_ue, type: real, use as feats for user nodes

        

        
        # node_features['user'] = np.array[(g.num_nodes('user') ,in_dim1)] ###############
        # node_features['antenna'] = np.array[(g.num_nodes('antenna') ,in_dim2)] #############
        # node_features['user'] = torch.tensor(g.num_nodes('user') ,in_dim1) ###############
        # node_features['antenna'] = torch.tensor(g.num_nodes('antenna') ,in_dim2) #############
        
        # edge_features = ...
        # g.edges[('user','channel','antenna')].data['feat'] = torch.tensor(int(g.num_edges()/2), in_dim3) #######

        # for node in g.n_types:
        #     g.nodes[node].data['feat'] = node_features[node]


        graphs.append(g)
    
    
    # valid_size = int(0.1*args.num_graphs) 
    # test_size = int(0.1*args.num_graphs) 

    # train_dict['psn_dev'] = np.array(psn_dev)[test_size+valid_size:]
    # train_dict['channel'] = np.array(channel)[test_size+valid_size:]
    # train_dict['pilots'] = np.array(pilots)[test_size+valid_size:]
    # train_dict['combiner'] = np.array(combiner)[test_size+valid_size:]

    # valid_dict['psn_dev'] = np.array(psn_dev)[test_size:test_size+valid_size]
    # valid_dict['channel'] = np.array(channel)[test_size:test_size+valid_size]
    # valid_dict['pilots'] = np.array(pilots)[test_size:test_size+valid_size]
    # valid_dict['combiner'] = np.array(combiner)[test_size:test_size+valid_size]

    # test_dict['psn_dev'] = np.array(psn_dev)[:test_size]
    # test_dict['channel'] = np.array(channel)[:test_size]
    # test_dict['pilots'] = np.array(pilots)[:test_size]
    # test_dict['combiner'] = np.array(combiner)[:test_size]

    # train_graphs = graphs[test_size+valid_size:]
    # valid_graphs = graphs[test_size:test_size+valid_size]
    # test_graphs = graphs[:test_size]

    data_dict['psn_dev'] = np.array(psn_dev)
    data_dict['channel'] = np.array(channel)
    data_dict['pilots'] = np.array(pilots)
    data_dict['combiner'] = np.array(combiner)


    # save the complete graphs
    # outfile = '/ubc/ece/home/ll/grads/idanroth/Desktop/eece_571f_project/data/dataset'
    dgl.save_graphs(os.path.join(path, args.filename +'.dgl'), graphs)

    # dgl.save_graphs(path + '_train.dgl', train_graphs)
    # dgl.save_graphs(path + '_valid.dgl', valid_graphs)
    # dgl.save_graphs(path + '_test.dgl', test_graphs)

    # save dataset
    np.save(os.path.join(path, args.filename +'.npy'), data_dict)
    # np.save(path + '_train', train_dict)
    # np.save(path + '_valid', valid_dict)
    # np.save(path + '_test', test_dict)

    t = (time.time() - start_time)
    h = int(np.floor(t/3600))
    m = int(np.floor((t-h*3600)/60))
    s = int(np.floor(t-h*3600-m*60))

    print(f"Succufully saved after: {h:0>2}:{m:0>2}:{s:0>2}")
    print(f"Example:")
    print(graphs[0])

    