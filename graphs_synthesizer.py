import os
os.environ["DGLBACKEND"] = "pytorch"
import dgl
import torch
import numpy as np
import argparse
import time
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fn', '--filename', type=str, help=".dgl file name of data, e.g., 'graph1'", required=True)
    parser.add_argument('-n', '--num_graphs', type=int, help="dataset size", default=440000)
    parser.add_argument('-tensor', '--tensor_flag', help="activate flag for the tensor form of the phase shifter network deviations", action="store_true")
    parser.add_argument('-tx', '--tx_flag', help="activate flag for the transmitter calibartion version", action="store_true")

    return parser.parse_args()



def generate_pilots(Nrf, Nr, Nue, P, dl, f, dev, tx_flag=False, dist="uniform", **tensor_params): 
    """ 
    Synthesize the recieved pilots at the base station where all users transmit simultaneously a symbol (using sitinct resources) via an uplink channel.
    Params:
        Nrf (int): number of radio frequency chains.
        Nr (int): number of antenna elements.
        Nue (int): number of users.
        P (Complex64 - [Nrf, Nr]): combiner matrix.
        dl (int): number of clusters for non line-of-sight channel scenario.
        f (int): carrier frequency.
        dev (int): the phase deviation "streangth" which correspond to a Uniform distrbution with a range [-dev,dev].
        tx_flag (bool): flag is True for transmitter calibration.
        dist (str): the phase deviation ditribution: 'uniform' or 'gaussian'.
    Returns:
        W (Complex64 - [Nrf, Nr]): PSN phase deviations.
        Y (Complex64 - [Nrf, Nue]): received pilots.
        H (Complex64 - [Nr, Nue]): channel.
    """

    if dist not in ["uniform","gaussian"]:
            raise KeyError(f'PSN distribution type {dist} not supported.\n'
                            'Use "uniform" or "gaussian" only')
    
    # Base station params
    wvlen = 3e8/f # wave length
    d = wvlen/2 # elements spacing

    if not tensor_params: # "Matrix" version

        if dist == "uniform": # Uniform PSN deviation
            W = np.deg2rad(np.random.uniform(-dev,dev, size=(Nrf,Nr)), dtype=np.float32)

        else: # Normal PSN deviations - estimation goal    
            sigma = np.deg2rad(dev)/(3**0.5) # To maintain the same std of a Uniform ditribution
            W = np.random.normal(0, sigma, size=(Nrf,Nr)).astype(dtype=np.float32) 
            
        W = np.exp(1j * W) # dtype = complex64, shape: Nrf X Nr 


        # Channel realizations
        # DoA
        a = np.array([np.arange(Nr)]) * (((2*d*np.pi)/wvlen)) # Constants of the steering vector
        doa = np.random.uniform(-90,90, size=(Nue,dl))

        A = np.exp(-1j * (np.transpose(a) * np.sin(np.deg2rad(doa.reshape(-1)))), dtype=np.complex64)

        # Gain
        G = np.zeros((Nue*dl, Nue), dtype=np.complex64)
        for n in range(Nue):
            re = np.random.normal(0, np.sqrt(1/2), dl)
            im = np.random.normal(0, np.sqrt(1/2), dl)
            g = re + 1j*im

            # g = np.exp(1j * np.random.uniform(-180,180, size=(dl))) #####################  Unit modolus with random phase  #####################
            G[n*dl:n*dl+dl, n] = g


        H = np.matmul(A, G) # dtype = complex64, shape: Nr X Nue
        Y = []
        Q = len(P)

        if not tx_flag: # Receiver calibration version
            for q in range(Q):
                # Generating recieved pilots according to system model
                Yq = np.matmul(np.multiply(W, P[q]), H) # dtype = complex64, shape: Nrf X Nue
                Y.append(Yq)
                
            # Stacking matrices vertically
            Y = np.array(Y).reshape(Q*Nrf, -1) # shape: Q*Nrf X Nue
        
        else: # Transmitter caliberation version
            for q in range(Q):
                # Generating recieved pilots according to system model
                s = np.ones((Nrf,1), dtype=np.complex64) # Trasnmitted pilot
                yq = np.matmul(H.conj().T, np.matmul(np.multiply(W, P[q]).conj().T, s)) # dtype = complex64, shape: Nue X 1
                Y.append(yq)
            
            # Stacking vectors vertically
            Y = np.array(Y)[:,:,-1] # shape: Q X Nue



    else: # Tensor version

        indices, Nb, correlation = tensor_params.values()

        if dist=="uniform":
            W = np.deg2rad(np.random.uniform(-dev,dev,size=(Nrf,Nr,Nb)), dtype=np.float32) # Uniform PSN deviations - estimation goal
        else: # Normal   
            if correlation is None:
                sigma = np.deg2rad(dev)/(3**0.5)
                W = np.random.normal(0, sigma, size=(Nrf,Nr,Nb)).astype(dtype=np.float32) # Normal PSN deviations - estimation goal
            else:
                dev = np.deg2rad(dev)
                sigma2 = (dev**2)/3 
                cov_mat = np.ones((Nb,Nb)) * sigma2 * correlation # Covariance matrix 
                np.fill_diagonal(cov_mat, sigma2) 

                W = np.random.multivariate_normal(np.zeros(Nb),cov_mat,size=Nrf*Nr).astype(dtype=np.float32).reshape(Nrf,Nr,Nb)

        W = np.exp(1j * W) # dtype = complex64, shape: Nrf X Nr X Nb
     
        psn_dev = np.transpose(W,[2,0,1]).reshape(Nb, Nrf*Nr) 


        # Channel realizations
        # DoA
        a = np.array([np.arange(Nr)]) * (((2*d*np.pi)/wvlen)) # Constants of the steering vector
        doa = np.random.uniform(-90,90, size=(Nue,dl))

        A = np.exp(-1j * (np.transpose(a) * np.sin(np.deg2rad(doa.reshape(-1)))), dtype=np.complex64)
        # Gain 
        G = np.zeros((Nue*dl, Nue), dtype=np.complex64)
        for n in range(Nue):
            re = np.random.normal(0, np.sqrt(1/2), dl)
            im = np.random.normal(0, np.sqrt(1/2), dl)
            g = re + 1j*im
            # g = np.exp(1j * np.deg2rad(np.random.uniform(-180,180,size=(dl))), dtype=np.complex64) # unit-modulus with random phase fading gain
            G[n*dl:n*dl+dl, n] = g

        H = np.matmul(A, G) # dtype = complex64, shape: Nr X Nue
        Y = []
        Q = len(P)

        if not tx_flag: # Receiver calibration version
            for q in range(Q):
                # Generating recieved pilots according to system model
                Wq = psn_dev[indices[q], np.arange(Nrf*Nr)].reshape(Nrf,Nr) # dtype = complex64, shape: Nrf X Nr
                Yq = np.matmul(np.multiply(Wq, P[q]), H) # dtype = complex64, shape: Nrf X Nue
                Y.append(Yq)
                
            # Stacking matrices vertically
            Y = np.array(Y).reshape(Q*Nrf, -1) # shape: Q*Nrf X Nue


    return  W, Y, H




if __name__ == "__main__":
    start_time = time.time()
    args = parse_args() # getting all the init args from input
    path = "<full_path_name>/data/" + args.filename # NOTE: need to replace <...> with the full path name of the directory which contain the script
    graph_data = {}

   
    # System model parameters
    if args.tensor_flag: # Tensor system model parameters
        f = 28e9 # transmission frequency
        Nr = 8 # no. of antenna elements
        Nue = 4 # no. of UEs
        Nrf = 1 # no. of RF chains
        phase_dev = 10 # +- range of phase deviation
        dl = 1 # no. of channel multipath
        snr_db = 5 # For no AWGN use the value: None
        B = 4 # B-bit phase shifter in the combiner
        O = 6
        Nb = 2**B
        Q = O*Nb # no. of measurements
        snr = 10**(snr_db/10)
        dist = 'gaussian' # phase deviation distribution: 'gaussian' or 'uniform'
        corr = None

    # Matrix system model parameters
    else:
        f = 28e9 # transmission frequency
        Nr = 16
        Nue = 4
        Nrf = 1 # no. of RF chains
        phase_dev = 15 # +- range of phase deviation under the Uniform distribution (or equivalent Gaussian distribution)
        dl = 1 # no. of channel multipath
        snr_db = 15 # For no AWGN use the value: None
        B = 5 # B-bit phase shifter in the combiner
        Q = 16 # no. of measurements
        snr = 10**(snr_db/10)
        dist = 'gaussian' # phase deviation distribution: 'gaussian' or 'uniform'
 

    print(f"System model parameters:\n no. antenna: {Nr}, no. users: {Nue}, no. RF chains: {Nrf}, no. measurements: {Q}, no. multipath: {dl}, SNR: {snr_db} dB, {B}-bit phase shifter")

    src1 = torch.arange(Nr).repeat_interleave(Nue)
    dst1 = torch.arange(Nue).repeat(Nr)
    src2 = torch.arange(Nue).repeat_interleave(Nr)
    dst2 = torch.arange(Nr).repeat(Nue)

    graph_data = {('antenna', 'channel2ue', 'user') : (src1, dst1),
                  ('user', 'channel2an', 'antenna') : (src2, dst2)}
        

    g = dgl.heterograph(graph_data, idtype=torch.int32)
    print(f"Graph feature types:\n nodes: {g.ntypes}, edges: {g.canonical_etypes}")

    graphs = []
    data_dict = {}

    data_dict['num_rfchain'] = Nrf
    data_dict['num_antenna'] = Nr
    data_dict['num_user'] = Nue
    data_dict['num_meas'] = Q
    data_dict['num_states'] = Nb if args.tensor_flag else None
    # data_dict['pilot_len'] = N if args.tensor_flag else None
    psn_dev, pilots, channel, combiner, indices = [],[],[],[],[]
    clean_pilots = []

    # Generating a random combiner for each measurement (fixed for all samples)
    # Using B-bit phase shifters
    delta = (2*np.pi) / (2**B)
    phases_list = np.linspace(-np.pi+delta, np.pi, 2**B, dtype=np.float32)
   
    # "matrix" version
    if not args.tensor_flag:
        idx = np.random.randint(0, 2**B, size=(Q,Nrf,Nr))

    # "tensor" version
    else:
        indices = np.zeros((Q, Nrf*Nr), dtype=np.uint32)
        for i in range(Nrf*Nr):
            indices[:,i] = np.random.permutation(Q) % Nb # Every column represent a phase shifter and holds a random permutation  
                                                         # of the values (0,1,...,N_B-1), each appearing O = Q/N_B times 

        idx = indices.reshape(Q,Nrf,Nr)

    
    P = phases_list[idx]
    combiner = np.exp(1j * P) # Analog combiner is unit modulus
    data_dict['indices'] = np.array(idx)


    for n in range(args.num_graphs):
 
        W, Y, H = generate_pilots(Nrf, Nr, Nue, combiner, dl, f, phase_dev, args.tx_flag, dist=dist, indices=indices, Nb=Nb, correlation=corr) if args.tensor_flag \
                  else generate_pilots(Nrf, Nr, Nue, combiner, dl, f, phase_dev, args.tx_flag, dist=dist) 
         
        psn_dev.append(W)
        pilots.append(Y) 
        channel.append(H)

        clean_pilots.append(Y.copy())


    pilots = np.array(pilots) # Shape: num_graph X Q*Nrf X Nue (Rx calibration) or num_graph X Q X Nue (Tx calibration)
    channel = np.array(channel) if not args.tx_flag else np.array(channel).conj() # Shape: num_graph X Nr X Nue
    psn_dev = np.array(psn_dev) if not args.tx_flag else np.array(psn_dev).conj() # Shape: num_graph X Nrf X Nt
    combiner = np.array(combiner).reshape(Q*Nrf, -1) if not args.tx_flag else np.array(combiner).conj().reshape(Q*Nrf, -1) # Stacking combiner matrices vertically, Shape: Q X Nrf X Nr --> Q*Nrf X Nr

    # Additive noise for Rx or Tx calibration
    if not args.tensor_flag:
        if snr_db is not None:
            power = np.linalg.norm(pilots, axis=(1,2), ord='fro')**2
            num_elem = Q*Nue if args.tx_flag else Q*Nrf*Nue
            sigma2_z = power.mean() / (snr*num_elem)

            size = (args.num_graphs, Q, Nue) if args.tx_flag else (args.num_graphs, Q*Nrf, Nue)
            re_z = np.random.normal(0, np.sqrt(sigma2_z/2), size).astype(np.float32)
            im_z = np.random.normal(0, np.sqrt(sigma2_z/2), size).astype(np.float32)
            Z = re_z + 1j*im_z
            pilots += Z

        
    
    # Graph node features
    user_feat = np.transpose(pilots, (0,2,1)) 
    user_feat = np.concatenate((user_feat.real, user_feat.imag), axis=2)
    # Concatenate CSI
    # channel_feat = np.transpose(channel, (0,2,1))
    # user_feat = np.concatenate((user_feat.real, user_feat.imag, channel_feat.real, channel_feat.imag), axis=2)

    antenna_feat = np.transpose(combiner, (1,0))
    antenna_feat = np.concatenate((antenna_feat.real, antenna_feat.imag), axis=1)

    # Data normalization
    mean, std = user_feat.mean(axis=0), user_feat.std(axis=0)
    user_feat = (user_feat-mean) / std

            

    for i in range(args.num_graphs):
        g = dgl.heterograph(graph_data, idtype=torch.int32)
        
        g.nodes['user'].data['feat'] = torch.tensor(user_feat[i]) # shape: Nue X 2*Q*Nrf (Rx calibrartion)  or   Nue X 2*Q (Tx calibration)
        g.nodes['antenna'].data['feat'] = torch.tensor(antenna_feat) # shape: Nr X 2*Q*Nrf

        # # Adding CSI as graph edge features for optional future use
        # H = channel[i] # shape: Nr X Nue
        # # Use transposed channel for channel2an because according to the defined graph data, the edges ordering is 0: (src0 to dst0), 1: (src0 to dst1),... 
        # # or src: [0,...,0,1...,1,2,...] to dst: [0,1,2,...,0,1,2,...]) where the src is a UE and the dst is the antenna, and because .reshape(-1,1) stacks row-wise
        # feat_channel2an = np.hstack([H.T.reshape(-1,1).real, H.T.reshape(-1,1).imag], dtype=np.float32) # shape: Nr*Nue X 2, order: h00, h01, h02,...,h0Nt, h10,... 
        # g.edges['channel2an'].data['e'] = torch.tensor(feat_channel2an) # use as feats for uplink edges
        # feat_channel2ue = np.hstack([H.reshape(-1,1).real, H.reshape(-1,1).imag], dtype=np.float32) # shape: Nue*Nr X 2, order: h00, h10, h20,...,hNue0, h01,... 
        # g.edges['channel2ue'].data['e'] = torch.tensor(feat_channel2ue) # use as feats for downlink edges    
    
        graphs.append(g)

    

    data_dict['psn_dev'] = psn_dev # Shape: D X Nrf X Nr
    data_dict['channel'] = channel # Shape: D X Nr X Nue
    data_dict['pilots'] = pilots # Shape: D X Q*Nrf X Nue
    data_dict['combiner'] = combiner # Shape: Q*Nrf X Nr

    data_dict['clean_pilots'] = np.array(clean_pilots)
  

    # save the complete graphs
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
            f.write(f'{"Tx" if args.tx_flag else "Rx"} calibration, {"non-" if not args.tensor_flag else ""}tensor version\n')
            f.write('Graph Parameters: \n')
            f.write(f'{graphs[0]} \n')
            f.write('System Parameters: \n')
            f.write(f'\t {B}-bit phase shifters \n')
            f.write(f'\t Fixed combiners for all samples \n')
            f.write(f'\t Antennas: {Nr}, users: {Nue}, rf-chains: {Nrf}, measurements: {Q}\n')
            f.write(f'\t SNR: {snr_db} dB, multipath: {dl}, Gaussian deviation: {phase_dev}, data size: {args.num_graphs}\n')

    print(f"Succufully saved after: {h:0>2}:{m:0>2}:{s:0>2}")
    print(f"Example:")
    print(graphs[0])

    