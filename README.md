# Graph Neural Network-Aided Online Calibration of Phase Shifter Networks

<br>

## Installation

Type the following command in the command line to create an Anaconda environement with the required Python libraries:

```bash
conda create --name <env> --file requirements.txt
```


## Files

### 1. graphs_synthesizer_w_fixed_combiner.py
Generate a dataset for the non-tensor version estimation.

Usage:
```bash
python graphs_synthesizer_w_fixed_combiner.py -fn <dataset name> -n <size>
```

Input flags: 
   
* -fn: dataset name
* -n: number of samples in the dataset (default: 320000)
* -e: add the CSI to the graph as edge features (default: True)
* -norm: normalize the dataset features (default: True)
* -tfeat: use the combiners phases as the antenna node features (default: True)

System parameters:
* main function:
    * Nt: no. of antenna
    * Nue: no. of users
    * Nrf: no. of RF chains
    * phase_dev: +- range of phase deviation under uniform distribution
    * dl: no. of channel multipath
    * snr_db: SNR value in dB for AWGN. use 'None' for zero noise
    * N: no. of measuremetns (note that in this version each pilot is a single symbol)
    * B: B-bit phase shifter in the combiner

* generate_pilots function:
    * f: transmission frequency
    * wvlen: wave length (= 3e8/f)
    * d: array elements spacing (= wvlen/2) 

Outputs:

save the following files in a folder named "-fn" under 'data' folder:

* .dgl file which hold the graph samples with their features.
* .npy file which holds the data dictionery with the following keys: "psn_dev", "pilots", "channel", "combiner", "num_rfchain" (no. of rf chains in the system), "num_meas" (no. of pilots measurements used in each sample), "num_states" (no. of phase states in each phase shifter). 
* .txt file which presents the parameters used for the dataset creation.

### 2. tensor_graphs_synthesizer.py
Generate a dataset for the tensor version estimation.

Usage:
```bash
python tensor_graphs_synthesizer.py -fn <dataset name> -n <size>
```

Input flags: 
   
* -fn: dataset name
* -n: number of samples in the dataset (default: 320000)
* -e: add the CSI to the graph as edge features (default: True)
* -norm: normalize the dataset features (default: True)
* -tfeat: use the combiners phases as the antenna node features (default: True)

System parameters:
* main function:
    * Nt: no. of antenna
    * Nue: no. of users
    * Nrf: no. of RF chains
    * phase_dev: +- range of phase deviation under uniform distribution
    * dl: no. of channel multipath
    * snr_db: SNR value in dB for AWGN. use 'None' for zero noise
    * N: pilot vector length
    * B: B-bit phase shifter in the combiner
    * O: no. of occurences for each phase state in every phase shifter
    * Nb: no. possible phase states (= 2**B)
    * Q: no. of measurements (= O*Nb)

* generate_pilots function:
    * f: transmission frequency
    * wvlen: wave length (= 3e8/f)
    * d: array elements spacing (= wvlen/2) 

Outputs:

save the following files in a folder named "-fn" under 'data' folder:

* .dgl file which hold the graph samples with their features.
* .npy file which holds the data dictionery with the following keys: "psn_dev", "pilots", "channel", "combiner", "num_rfchain" (no. of rf chains in the system), "num_meas" (no. of pilots measurements used in each sample), "num_states" (no. of phase states in each phase shifter). 
* .txt file which presents the parameters used for the dataset creation.

### 3. train.py
Train a choosen GNN model type based on the given dataset, save the model under 'models ' file, and print out the RMSE test result over the testset (which is also being saved in the dataset folder). 

Usage:
```bash
python train.py -fn <dataset name> -csi <bool> -t <bool>
```

Input flags: 
   
* -fn: dataset name
* -s: if want to specify a name for saving the model
* -opt: optimizaer type: 'ADAM' or 'SGD' (default='ADAM')
* -act: activation function: 'ReLU', 'LeakyReLU' or 'ELU' (default='ReLU')
* -mod: wandb operation mode, use 'disabled' for debugging (default='online')
* -csi: if True, CSI would be used in the GNN (default=False)
* -t: if True, the tensor version GNN of the PSN estimation would be used (default=True)

Parameters:

* num_epochs: no. of epochs to train the model
* batch_size: minibatch size for the dataset iterator
* learning_rate: learning rate of the optimizaer
* betas: values for ADAM optimizer
* weight_decay: L2 penalty coefficient
* dropout: dropout probability for the MLP used in the GNN
* d_h: MLP hidden dimension for the nodes hidden representation
* n_layer: no. of layers used in the MLPs of the GNN

### 4. models.py
File which contains four Pytorch models and the classes that form them:

* GraphNeuralNetworkConcat2 - the baseline GNN which does not uses CSI as part of training. It is constructed using an intilaization layer, graph convolution layers, and a normalization layer.
* SkipConGnn - same as the baseline GNN but uses for training skip connections from each GCN layer output.
* GraphNeuralNetworkCSIConcat -  GCN which concatenates the CSI features to the recieved pilots for the UE nodes hidden representation in the intialization layer (1st version of CSI GCN).
* EdgeGraphNeuralNetwork - GCN which uses CSI as features on edges in the aggregation stage (2nd version of CSI GCN).

Moreover, the file contains training loop functions (+ validation) and testing loop functions, which are used by 'train.py' to train both GNN veresions (non-tensor and tensor).

### 5. utils.py
File which contains utilities functions such as Pytorch dataset objects, save function, RMSE loss function, etc.


## Usage

1. Create a dataset by running the tensor or non-tensor graph sythesizer option, after specifying the system parameters in the code. Run e.g.:
```bash
python tensor_graphs_synthesizer.py -fn 'tensor_graph_fixed_2feat_20db_03Jan24'
```
2. Train the model after specifying the hyperparameters in the code. Adjust the flags used in the command for the appropriate data type (tensor or non-tensor) and GNN option (w/ CSI or w/o CSI), test result would be printed on terminal. Run e.g.:

```bash
python train.py -fn 'tensor_graph_fixed_2feat_20db_03Jan24' -csi True
```

    Note: wandb is used in code for training vizualization by logging results to their servers.



#### Code was written by Idan Roth, idanroth@ece.ubc.ca, under the supervision of Prof. Lutz Lampe from the ECE department at UBC.
