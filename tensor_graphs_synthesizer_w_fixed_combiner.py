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
    parser.add_argument('-n', '--num_graphs', type=int, help="dataset size", default=320000)
    parser.add_argument('-e', '--edge_feat', type=bool, help="True to use channel as edge features, False otherwise", default=False)
    parser.add_argument('-norm', '--norm', type=bool, help="True to normalize dataset", default=True)
    parser.add_argument('-tfeat', '--antenna_feat', type=bool, help="True to use combiner as antenna node features", default=True)

    return parser.parse_args()



def generate_pilots(Nrf, Nt, Nue, Nb, P, indices, dl, dev):
    # Base station params
    # Base station params
    f = 28e9 # transmission frequency
    wvlen = 3e8/f # wave length
    d = wvlen/2 # elements spacing


    W = np.deg2rad(np.random.uniform(-dev,dev,size=(Nrf,Nt,Nb)), dtype=np.float32) # PSN deviations - estimation goal
    W = np.exp(1j * W) # dtype = complex64, shape: Nrf X Nt X Nb
    psn_dev = np.transpose(W,[2,0,1]).reshape(Nb, Nrf*Nt)

    # Channel realizations
    # DoA
    a = np.array([np.arange(Nt)]) * (((2*d*np.pi)/wvlen)) # Constants of the steering vector
    doa = np.zeros((Nue, dl))
    doa[:,] = np.random.uniform(-90,90,size=(Nue,1)) # Duplicate DoA to all columns 
    spread = np.random.uniform(-10,10,size=(Nue,dl-1))
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

    Y = []
    N = len(P)
    for n in range(N):
 
        Wn = psn_dev[indices[n], np.arange(Nrf*Nt)].reshape(Nrf,Nt) # dtype = complex64, shape: Nrf X Nt

        # Generating the recieved pilots according to system model
        Yn = np.matmul(np.multiply(Wn, P[n]), H) # dtype = complex64, shape: Nrf X Nue
        Y.append(Yn)
        
    
    # Stacking matrices vertically
    Y = np.array(Y).reshape(N*Nrf, -1) # shape: N*Nrf X Nue

    return  W, Y, H



if __name__ == "__main__":
    start_time = time.time()
    args = parse_args() # getting all the init args from input
    path = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/" + args.filename

    # System model parameters
    Nt = 8 # no. of antenna elements
    Nue = 3 # no. of UEs
    Nrf = 2 #Nt//8 # no. of RF chains
    phase_dev = 20 # +- range of phase deviation
    dl = 1 # no. of channel multipath
    snr_db = 20 # For no AWGN use the value: None
    B = 4 # B-bit phase shifter in the combiner
    O = 10
    Nb = 2**B
    Q = O*Nb # no. of measurements

    src1 = torch.arange(Nt).repeat_interleave(Nue)
    dst1 = torch.arange(Nue).repeat(Nt)
    src2 = torch.arange(Nue).repeat_interleave(Nt)
    dst2 = torch.arange(Nt).repeat(Nue)

    graph_data = {('antenna', 'channel2u', 'user'): (src1, dst1),
                  ('user', 'channel2a', 'antenna'): (src2, dst2)}

    g = dgl.heterograph(graph_data, idtype=torch.int32)
    print(f"Graph feature types:\n nodes: {g.ntypes}, edges: {g.canonical_etypes}")

    graphs = []
    data_dict = {}

    print(f"System model parameters:\n no. antenna: {Nt}, no. users: {Nue}, no. RF chains: {Nrf}, no. measurements: {Q}, no. multipath: {dl}, SNR: {snr_db} dB, {B}-bit phase shifter")

    data_dict['num_rfchain'] = Nrf
    data_dict['num_meas'] = Q
    data_dict['num_states'] = Nb
    # data_dict['num_antenna'] = Nt
    # data_dict['num_user'] = Nue
    psn_dev, pilots, channel, combiner = [],[],[],[]

    # Generating a random combiner for each measurement (fixed for all samples)
    # Using B-bit phase shifters
    delta = (2*np.pi) / (2**B)
    phases_list = np.linspace(-np.pi+delta, np.pi, 2**B, dtype=np.float32)
    idx = np.zeros((Q, Nrf*Nt), dtype=np.uint32)

    for i in range(Nrf*Nt):
        idx[:,i] = np.random.permutation(Q) % Nb # Permuted selection for each phase shifter

    # for q in range(Q):
    #     Pq = phases_list[idx[q].reshape(Nrf,Nt)]
    #     Pq = np.exp(1j * Pq) # Analog combiner is unit modulus
    #     combiner.append(Pq)
    
    P = phases_list[idx.reshape(Q,Nrf,Nt)]
    combiner = np.exp(1j * P)

    # if args.norm:
    for n in range(args.num_graphs):
        W, Y, H = generate_pilots(Nrf, Nt, Nue, Nb, combiner, idx, dl, phase_dev)
        psn_dev.append(W)
        pilots.append(Y)
        channel.append(H)

    pilots = np.array(pilots)
    # Additive noise
    if snr_db is not None:
        snr = 10**(snr_db/10)
        power = np.linalg.norm(pilots, axis=(1,2), ord='fro')**2
        sigma2_z = power.mean() / (snr*Q*Nrf*Nue)

        for n in range(args.num_graphs):
            re_z = np.random.normal(0, np.sqrt(sigma2_z/2), (Q*Nrf,Nue)).astype(np.float32)
            im_z = np.random.normal(0, np.sqrt(sigma2_z/2), (Q*Nrf,Nue)).astype(np.float32)
            Z = re_z + 1j*im_z
            pilots[n] += Z

    feat = np.transpose(pilots, (0,2,1))
    feat = np.concatenate((feat.real, feat.imag), axis=2)

    if args.antenna_feat:
        antenna_feat = np.transpose(combiner.reshape(Q*Nrf, -1), (1,0))
        antenna_feat = np.concatenate((antenna_feat.real, antenna_feat.imag), axis=1)

    # Data normalization
    if args.norm:
        mean, std = feat.mean(axis=0), feat.std(axis=0)
        feat = (feat-mean) / std

        # if args.antenna_feat:
        #     mean, std = antenna_feat.mean(axis=0), antenna_feat.std(axis=0)
        #     antenna_feat = (antenna_feat-mean) / std

    for i in range(args.num_graphs):
        g = dgl.heterograph(graph_data, idtype=torch.int32)
        # create graph features
        # Add node feats only for user nodes! 
        # Antenna node feats would be embedded using the user feats inside the gnn model
        g.nodes['user'].data['feat'] = torch.tensor(feat[i]) # shape: Nue X 2*N*Nrf
        if args.antenna_feat:
            g.nodes['antenna'].data['feat'] = torch.tensor(antenna_feat) # shape: Nt X 2*N*Nrf
        graphs.append(g)

    

    data_dict['psn_dev'] = np.array(psn_dev)
    data_dict['channel'] = np.array(channel)
    data_dict['pilots'] = np.array(pilots)
    data_dict['combiner'] = np.array(combiner).reshape(Q*Nrf, -1) # Stacking combiner matrices vertically


    # save the complete graphs
    # outfile = '/ubc/ece/home/ll/grads/idanroth/Desktop/eece_571f_project/data/dataset'
    dgl.save_graphs(os.path.join(path, args.filename +'.dgl'), graphs)

    # save dataset
    np.save(os.path.join(path, args.filename +'.npy'), data_dict)


    t = (time.time() - start_time)
    h = int(np.floor(t/3600))
    m = int(np.floor((t-h*3600)/60))
    s = int(np.floor(t-h*3600-m*60))

    if not os.path.isfile(path + '/params.txt'):
        with open( path + '/params.txt', 'w') as f:
            f.write('Graph Parameters \n')
            f.write(f'\t {B}-bit phase shifters with +-{phase_dev} dev\n')
            f.write(f'\t Fixed combiners for all samples \n')
            if args.antenna_feat:
                f.write(f'\t Both node type features were intialized \n')
            if args.edge_feat:
                f.write(f'\t Channel as uplink edge features \n')
            f.write(f'\t Antennas: {Nt}, users: {Nue}, rf-chains: {Nrf}, measurements: {Q}\n')
            f.write(f'\t SNR: {snr_db} dB, multipath: {dl}, data size: {args.num_graphs}\n')

    print(f"Succufully saved after: {h:0>2}:{m:0>2}:{s:0>2}")
    print(f"Example:")
    print(graphs[0])
    

    