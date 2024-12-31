import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import glob
import numpy as np
import pandas as pd
import argparse
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--filename', type=str, help=".dgl file name of data, e.g., 'graph1'", required=True)
    parser.add_argument('-n', '--num_graphs', type=int, help="dataset size", default=320000)
    parser.add_argument('-e', '--edge_feat', type=bool, help="True to add channel data in graph edge features, False otherwise", default=True)
    parser.add_argument('-norm', '--norm', type=bool, help="True to normalize dataset", default=True)
    # parser.add_argument('-tfeat', '--antenna_feat', type=bool, help="True to use combiner as antenna node features", default=True)
    parser.add_argument('-e_gnn', '--edge_gnn', help="True to use combiner as antenna node features", action="store_true")
    parser.add_argument('-e_reg', '--edge_regressor', help="True to use combiner as antenna node features", action="store_true")

    return parser.parse_args()



def generate_pilots(Nrf, Nt, Nue, P, dl, f, dev):
    # Base station params
    wvlen = 3e8/f # wave length
    d = wvlen/2 # elements spacing


    W = np.deg2rad(np.random.uniform(-dev,dev,size=(Nrf,Nt)), dtype=np.float32) # PSN deviations - estimation goal
    W = np.exp(1j * W) # dtype = complex64, shape: Nrf X Nt

    # Eliminate phase shift ambiguity
    W[0,0] = 1 # implies the phase to be zero
    

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
    Q = len(P)
    for q in range(Q):
        # Generating recieved pilots according to system model
        Yq = np.matmul(np.multiply(W, P[q]), H) # dtype = complex64, shape: Nrf X Nue
        Y.append(Yq)
        
    
    # Stacking matrices vertically
    Y = np.array(Y).reshape(Q*Nrf, -1) # shape: Q*Nrf X Nue

    return  W, Y, H



if __name__ == "__main__":
    start_time = time.time()
    args = parse_args() # getting all the init args from input
    path = "/ubc/ece/home/ll/grads/idanroth/Projects/gnn_psn_calib/data/" + args.filename
    graph_data = {}

    # for fname in glob.glob(path + "/*.csv"):
    #     df = pd.read_csv(fname)

    #     src = torch.tensor(df['src'].to_numpy())
    #     dst = torch.tensor(df['dst'].to_numpy())
    #     # get nodes names and relations from fname
    #     canonical_etype = tuple(os.path.basename(os.path.splitext(fname)[0]).split('-'))
    #     # build a relation dictionary 
    #     graph_data[canonical_etype] = (src, dst)

   
    # System model parameters
    # Nt = g.num_nodes('antenna') # no. of antenna elements
    f = 28e9 # transmission frequency
    phase_dev = 10 # +- range of phase deviation
    Nt = 16
    # Nue = g.num_nodes('user') # no. of UEs
    Nue = 4
    Nrf = 2 #Nt//8 # no. of RF chains
    Q = 16#Nt # no. of measurements
    dl = 1 # no. of channel multipath
    snr_db = 20 # For no AWGN use the value: None
    B = 5 # B-bit phase shifter in the combiner

    print(f"System model parameters:\n no. antenna: {Nt}, no. users: {Nue}, no. RF chains: {Nrf}, no. measurements: {Q}, no. multipath: {dl}, SNR: {snr_db} dB, {B}-bit phase shifter")

    if args.edge_regressor:
        # '2rel'
        src1 = torch.arange(Nrf).repeat_interleave(Nt)
        dst1 = torch.arange(Nt).repeat(Nrf)
        src2 = torch.arange(Nt).repeat_interleave(Nrf)
        dst2 = torch.arange(Nrf).repeat(Nt)
        # + including the following will give '4rel'
        src3 = torch.arange(Nt).repeat_interleave(Nue)
        dst3 = torch.arange(Nue).repeat(Nt)
        src4 = torch.arange(Nue).repeat_interleave(Nt)
        dst4 = torch.arange(Nt).repeat(Nue)
        # + including the following will give '6rel'
        src5 = torch.arange(Nue).repeat_interleave(Nrf)
        dst5 = torch.arange(Nrf).repeat(Nue)
        src6 = torch.arange(Nrf).repeat_interleave(Nue)
        dst6 = torch.arange(Nue).repeat(Nrf)
        
        graph_data = {('rfchain', 'psn', 'antenna')  : (src1, dst1),
                      ('antenna', 'psn', 'rfchain')  : (src2, dst2),
                      ('antenna', 'channel', 'user') : (src3, dst3),
                      ('user', 'channel', 'antenna') : (src4, dst4),
                      ('user', 'pilot', 'rfchain')   : (src5, dst5),
                      ('rfchain', 'pilot', 'user')   : (src6, dst6)}
    
    elif args.edge_gnn:
        src1 = torch.arange(Nt).repeat_interleave(Nue)
        dst1 = torch.arange(Nue).repeat(Nt)
        src2 = torch.arange(Nue).repeat_interleave(Nt)
        dst2 = torch.arange(Nt).repeat(Nue)
        src3 = torch.arange(Nrf).repeat_interleave(Nt)
        dst3 = torch.arange(Nt).repeat(Nrf)
        src4 = torch.arange(Nt).repeat_interleave(Nrf)
        dst4 = torch.arange(Nrf).repeat(Nt)
        src5 = torch.arange(Nue).repeat_interleave(Nrf)
        dst5 = torch.arange(Nrf).repeat(Nue)
        src6 = torch.arange(Nrf).repeat_interleave(Nue)
        dst6 = torch.arange(Nue).repeat(Nrf)

        graph_data = {('antenna', 'channel', 'user'): (src1, dst1),
                      ('user', 'channel', 'antenna'): (src2, dst2),
                      ('rfchain', 'psn', 'antenna'): (src3, dst3),
                      ('antenna', 'psn', 'rfchain'): (src4, dst4),
                      ('user', 'pilot', 'rfchain'): (src5, dst5),
                      ('rfchain', 'pilot', 'user'): (src6, dst6)}

    else: # first version
        src1 = torch.arange(Nt).repeat_interleave(Nue)
        dst1 = torch.arange(Nue).repeat(Nt)
        src2 = torch.arange(Nue).repeat_interleave(Nt)
        dst2 = torch.arange(Nt).repeat(Nue)

        graph_data = {('antenna', 'channel2ue', 'user'): (src1, dst1),
                      ('user', 'channel2an', 'antenna'): (src2, dst2)}
        

    g = dgl.heterograph(graph_data, idtype=torch.int32)
    print(f"Graph feature types:\n nodes: {g.ntypes}, edges: {g.canonical_etypes}")

    graphs = []
    data_dict = {}


    data_dict['num_rfchain'] = Nrf
    data_dict['num_meas'] = Q
    data_dict['num_antenna'] = Nt
    data_dict['num_user'] = Nue
    psn_dev, pilots, channel, combiner = [],[],[],[]
    indices = []

    # Generating a random combiner for each measurement (fixed for all samples)
    # Using B-bit phase shifters
    delta = (2*np.pi) / (2**B)
    phases_list = np.linspace(-np.pi+delta, np.pi, 2**B, dtype=np.float32)
    # when using a fixed combiner set
    indices = []
    for i in range(Q):
        idx = np.random.randint(0, 2**B, size=(Nrf,Nt))
        Pn = phases_list[idx]
        # Pn = np.deg2rad(np.random.randint(-180,180,size=(Nrf,Nt)), dtype=np.float32) 
        Pn = np.exp(1j * Pn) # Analog combiner is unit modulus
        combiner.append(Pn)
        indices.append(idx)
    data_dict['indices'] = np.array(indices)

    # if args.norm:
    for n in range(args.num_graphs):
        # when not using a fixed combiner set
        # curr_comb = []
        # for i in range(Q):
        #     idx = np.random.randint(0, 2**B, size=(Nrf,Nt))
        #     Pn = phases_list[idx]
        #     # Pn = np.deg2rad(np.random.randint(-180,180,size=(Nrf,Nt)), dtype=np.float32) 
        #     Pn = np.exp(1j * Pn) # Analog combiner is unit modulus
        #     curr_comb.append(Pn)

        W, Y, H = generate_pilots(Nrf, Nt, Nue, combiner, dl,f, phase_dev)
        psn_dev.append(W)
        pilots.append(Y) 
        channel.append(H)
        # when not using a fied combiner set
        # combiner.append(np.array(curr_comb))

    pilots = np.array(pilots) # Shape: num_graph X Q*Nrf X Nu
    channel = np.array(channel)
    combiner = np.array(combiner)
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

        
    if args.edge_gnn:
        # Arranging edge features to the proper shape
        # Channel edge features
        H = channel # shape: D X Nt X Nue (D is the dataset size)
        # Use transposed channel for channel2an because according to the defined graph data, the edges ordering is 0: (src0 to dst0), 1: (src0 to dst1),... 
        # or src: [0,...,0,1...,1,2,...] to dst: [0,1,2,...,0,1,2,...]) where the src is a UE and the dst is the antenna, and because .reshape(-1,1) stacks row-wise
        H_2an = H.transpose(0,2,1).reshape(args.num_graphs,-1,1) # shape: D X Nt*Nue X 1
        feat_channel2an = np.concatenate([H_2an.real, H_2an.imag], axis=2, dtype=np.float32) # shape: D X Nt*Nue X 2, order: h00, h01, h02,...,h0Nt, h10,... 
        H_2ue = H.reshape(args.num_graphs,-1,1)
        feat_channel2ue = np.concatenate([H_2ue.real, H_2ue.imag], axis=2, dtype=np.float32) # shape: D X Nue*Nt X 2, order: h00, h10, h20,...,hNue0, h01,... 

        # pilots edge features
        Y = pilots.reshape(args.num_graphs,Q,Nrf,Nue) # shape: D X Q X Nrf X Nue
        Y_2rf = Y.transpose(0,1,3,2).reshape(args.num_graphs,Q,-1).transpose(0,2,1) # shape: D X Nrf*Nue X Q
        feat_pilot2rf = np.concatenate([Y_2rf.real, Y_2rf.imag], axis=2 , dtype=np.float32) # shape: D X Nue*Nrf X 2*Q
        Y_2an = Y.reshape(args.num_graphs,Q,-1).transpose(0,2,1) # shape: D X Nrf*Nue X Q
        feat_pilot2ue = np.concatenate([Y_2an.real, Y_2an.imag], axis=2 , dtype=np.float32) # shape: D X Nue*Nrf X 2*Q

        # PSN edge features (combiner) - when not using a fixed combiner set
        # P = combiner # shape: D X Q X Nrf X Nt 
        # P_2an = P.reshape(args.num_graphs,Q,-1).transpose(0,2,1) # shape: D X Nrf*Nt X Q
        # feat_psn2an = np.concatenate([P_2an.real, P_2an.imag], axis=2 , dtype=np.float32) # shape: D X Nt*Nrf X 2*Q
        # P_2rf = P.transpose(0,1,3,2).reshape(args.num_graphs,Q,-1).transpose(0,2,1) # shape: D X Nrf*Nt X Q
        # feat_psn2rf = np.concatenate([P_2rf.real, P_2rf.imag], axis=2 , dtype=np.float32) # shape: D X Nt*Nrf X 2*Q
    

        if args.norm:
            # Data normalization over samples 
            feat_channel2an = (feat_channel2an - feat_channel2an.mean(axis=0)) / feat_channel2an.std(axis=0)
            feat_channel2ue = (feat_channel2ue - feat_channel2ue.mean(axis=0)) / feat_channel2ue.std(axis=0)
            feat_pilot2rf = (feat_pilot2rf - feat_pilot2rf.mean(axis=0)) / feat_pilot2rf.std(axis=0)
            feat_pilot2ue = (feat_pilot2ue - feat_pilot2ue.mean(axis=0)) / feat_pilot2ue.std(axis=0)
            # when not using a fixed combiner set
            # feat_psn2an = (feat_psn2an - feat_psn2an.mean(axis=0)) / feat_psn2an.std(axis=0)
            # feat_psn2rf = (feat_psn2rf - feat_psn2rf.mean(axis=0)) / feat_psn2rf.std(axis=0)

    elif args.edge_regressor:
        antenna_feat = np.transpose(np.copy(combiner).reshape(Q*Nrf, -1), (1,0))
        antenna_feat = np.concatenate((antenna_feat.real, antenna_feat.imag), axis=1)

        rfchain_feat = np.transpose(np.copy(pilots).reshape(args.num_graphs, Q, Nrf, Nue), (0,1,3,2))
        rfchain_feat = np.transpose(rfchain_feat.reshape(args.num_graphs, Q*Nue, Nrf), (0,2,1))
        rfchain_feat = np.concatenate((rfchain_feat.real, rfchain_feat.imag), axis=2)

        if len(g.canonical_etypes) != 2: # '4rel' or '6rel'
            user_feat = np.transpose(np.copy(pilots), (0,2,1))
            user_feat = np.concatenate((user_feat.real, user_feat.imag), axis=2)
        # Concatenate CSI
        # channel_feat = np.transpose(channel, (0,2,1))
        # user_feat = np.concatenate((user_feat.real, user_feat.imag, channel_feat.real, channel_feat.imag), axis=2)

        # Data normalization
        if args.norm:
            mean, std = rfchain_feat.mean(axis=0), rfchain_feat.std(axis=0)
            rfchain_feat = (rfchain_feat-mean) / std
            if len(g.canonical_etypes) != 2: # '4rel' or '6rel'
                mean, std = user_feat.mean(axis=0), user_feat.std(axis=0)
                user_feat = (user_feat-mean) / std
    
    else: # first version
        user_feat = np.transpose(pilots, (0,2,1))
        user_feat = np.concatenate((user_feat.real, user_feat.imag), axis=2)
        # Concatenate CSI
        # channel_feat = np.transpose(channel, (0,2,1))
        # user_feat = np.concatenate((user_feat.real, user_feat.imag, channel_feat.real, channel_feat.imag), axis=2)

        antenna_feat = np.transpose(combiner.reshape(Q*Nrf, -1), (1,0))
        antenna_feat = np.concatenate((antenna_feat.real, antenna_feat.imag), axis=1)
    
        # Data normalization
        if args.norm:
            mean, std = user_feat.mean(axis=0), user_feat.std(axis=0)
            user_feat = (user_feat-mean) / std

        # if args.antenna_feat:
        #     mean, std = antenna_feat.mean(axis=0), antenna_feat.std(axis=0)
        #     antenna_feat = (antenna_feat-mean) / std
            

    for i in range(args.num_graphs):
        g = dgl.heterograph(graph_data, idtype=torch.int32)
            
        if args.edge_gnn:
        # Channel edge features
            g.edges[('user', 'channel', 'antenna')].data['e'] = torch.tensor(feat_channel2an[i]) # use as feats for uplink edges (feature vecotr of size 2)
            g.edges[('antenna', 'channel', 'user')].data['e'] = torch.tensor(feat_channel2ue[i]) # use as feats for downlink edges (feature vecotr of size 2)
        
        # PSN edge features (combiner)
            # When using a fixed combiner set
            psn = np.array(combiner) # shape: Q X Nrf X Nt 
            feat_psn2an = np.hstack([psn.reshape(Q,-1).T.real, psn.reshape(Q,-1).T.imag], dtype=np.float32) # shape: Nt*Nrf X 2*Q
            g.edges[('rfchain', 'psn', 'antenna')].data['e'] = torch.tensor(feat_psn2an) # (feature vecotr of size 2*Q)
            feat_psn2rf = np.hstack([psn.transpose(0,2,1).reshape(Q,-1).T.real, psn.transpose(0,2,1).reshape(Q,-1).T.imag], dtype=np.float32) # shape: Nt*Nrf X 2*Q
            g.edges[('antenna', 'psn', 'rfchain')].data['e'] = torch.tensor(feat_psn2rf) # (feature vecotr of size 2*Q)

            # When not using a fixed combiner set
            # # g.edges['psn2an'].data['e'] = torch.tensor(feat_psn2an[i]) # (feature vecotr of size 2*Q)
            # # g.edges['psn2rf'].data['e'] = torch.tensor(feat_psn2rf[i]) # (feature vecotr of size 2*Q)
            # g.edges[('rfchain', 'psn', 'antenna')].data['e'] = torch.tensor(feat_psn2an[i]) # (feature vecotr of size 2*Q)
            # g.edges[('antenna', 'psn', 'rfchain')].data['e'] = torch.tensor(feat_psn2rf[i]) # (feature vecotr of size 2*Q)
            

        # pilots edge features
            g.edges[('user', 'pilot', 'rfchain')].data['e'] = torch.tensor(feat_pilot2rf[i]) # (feature vecotr of size 2*Q)
            g.edges[('rfchain', 'pilot', 'user')].data['e'] = torch.tensor(feat_pilot2ue[i]) # (feature vecotr of size 2*Q)
         
        
        elif args.edge_regressor:
            g.nodes['rfchain'].data['feat'] = torch.tensor(rfchain_feat[i]) # shape: Nrf X 2*Q*Nue
            g.nodes['antenna'].data['feat'] = torch.tensor(antenna_feat) # shape: Nt X 2*Q*Nrf
            
            if len(g.canonical_etypes) != 2: # '4rel' or '6rel'
                g.nodes['user'].data['feat'] = torch.tensor(user_feat[i]) # shape: Nue X 2*Q*Nrf
        
        else:
            g.nodes['user'].data['feat'] = torch.tensor(user_feat[i]) # shape: Nue X 2*Q*Nrf
            g.nodes['antenna'].data['feat'] = torch.tensor(antenna_feat) # shape: Nt X 2*Q*Nrf

            if args.edge_feat:
                H = channel[i] # shape: Nt X Nue
                # Use transposed channel for channel2an because according to the defined graph data, the edges ordering is 0: (src0 to dst0), 1: (src0 to dst1),... 
                # or src: [0,...,0,1...,1,2,...] to dst: [0,1,2,...,0,1,2,...]) where the src is a UE and the dst is the antenna, and because .reshape(-1,1) stacks row-wise
                feat_channel2an = np.hstack([H.T.reshape(-1,1).real, H.T.reshape(-1,1).imag], dtype=np.float32) # shape: Nt*Nue X 2, order: h00, h01, h02,...,h0Nt, h10,... 
                g.edges['channel2an'].data['e'] = torch.tensor(feat_channel2an) # use as feats for uplink edges
                feat_channel2ue = np.hstack([H.reshape(-1,1).real, H.reshape(-1,1).imag], dtype=np.float32) # shape: Nue*Nt X 2, order: h00, h10, h20,...,hNue0, h01,... 
                g.edges['channel2ue'].data['e'] = torch.tensor(feat_channel2ue) # use as feats for downlink edges    
    
        graphs.append(g)

    

    data_dict['psn_dev'] = np.array(psn_dev) # Shape: D X Nrf X Nt
    data_dict['channel'] = np.array(channel) # Shape: D X Nt X Nue
    data_dict['pilots'] = np.array(pilots) # Shape: D X Q*Nrf X Nue
    # when combiner is fixed:
    data_dict['combiner'] = np.array(combiner).reshape(Q*Nrf, -1) # Stacking combiner matrices vertically, Shape: Q*Nrf X Nt
    # when combiner is not fixed:
    # data_dict['combiner'] = np.array(combiner.reshape(args.num_graphs, Q*Nrf, -1))


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
            f.write(f'Date: {datetime.now().strftime("%d-%m-%Y-@-%H:%M")} \n')
            f.write('Graph Parameters: \n')
            # if args.first_version:
            #     f.write(f'\t Both node type features were intialized \n')
                # if args.edge_feat:
                #     f.write(f'\t Channel as uplink edge features \n')
            if args.edge_regressor:
                f.write(f"Using {len(g.canonical_etypes)} relations in graph \n")
            
            f.write(f'{graphs[0]} \n')
            f.write('System Parameters: \n')
            f.write(f'\t {B}-bit phase shifters \n')
            # if args.fixed_comb:
            f.write(f'\t Fixed combiners for all samples \n')
            f.write(f'\t Antennas: {Nt}, users: {Nue}, rf-chains: {Nrf}, measurements: {Q}\n')
            f.write(f'\t SNR: {snr_db} dB, multipath: {dl}, data size: {args.num_graphs}\n')

    print(f"Succufully saved after: {h:0>2}:{m:0>2}:{s:0>2}")
    print(f"Example:")
    print(graphs[0])

    