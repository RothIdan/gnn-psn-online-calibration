# Graph Neural Network-Aided Online Calibration of Phase Shifter Networks

## Introduction

This repository contains the implementation code used in the research described in the journal paper:

"Roth, I., & Lampe, L. (2025). Graph Neural Network-Aided Online Calibration of Phase Shifter Networks."

The code provided herein is intended to facilitate the replication of results and further research in the area of graph neural networks for online calibration of phase shifter networks. It was used for simulations that are demonstrated and discussed in our paper, which is currently under review.

**Author**: This code was written by Idan Roth to support the research presented in the paper.

**Prerequisites**: For details on the software environment and libraries required to run the simulations, please refer to the **Requirements and Installation** section and the `requirements.txt` file included in this repository.

For any further questions or collaboration inquiries, please contact: idanroth@ece.ubc.ca.

<br>

## Requirements and Installation

Run the following command to create an Anaconda environement with the required Python packages:

```bash
conda create --name <str> --file requirements.txt
```
or
```bash
conda env create -f environment.yml
```


## Detailed Code Overview and Usage

### 1. graphs_synthesizer.py
Generate the dataset in two files: .dgl file which contains the graphs with their features, and .npy file which contains the dictionary of relevant generated data and parameters that were used for the graphs dataset. Outputs also a .txt file with the specific parameters that were used.

Usage:
```bash
python graphs_synthesizer.py -fn <str> -n <int>
```

Input flags: 
   
* -fn: directory name for saving dataset in (will be saved inside the 'data' directory)
* -n: number of samples in the dataset (default: 320000)
* -tensor: flag to activate the tensor version of the phase shifter network deviations
* -tx: flag to activate the transmitter calibartion version


main function:
* Need to set the desired system parameters inside the main function for the matrix form or the tensor form of the system model.
* Inputs:
    * Nr: number of antennas
    * Nue: number of users
    * Nrf: number of RF chains
    * phase_dev: +- range of phase deviation under uniform distribution (or for the equivalent Gaussian distribution s.t.d.)
    * dl: number of channel clusters
    * snr_db: SNR value in dB for AWGN
    * B: B-bit phase shifter in the combiner
    * Q: number of training combiners (measuremetns) for the matrix form
    * O: number of phase states appearances for the tensor form

    Note: need to replace '<...>' in varaible 'path' with the full path name of the directory which contain this script
   

Outputs:

Save the following files in a directory named "-fn" under "data" directory:

* .dgl file which hold the graph samples with their features.
* .npy file which holds the data dictionery with the following keys: "psn_dev", "pilots", "channel", "combiner", "num_rfchain", "num_antenna", "num_meas", "num_states, "num_user. 
* .txt file which presents the system parameters used for the dataset creation.


### 2. train.py
Train a choosen GNN model type based on the given dataset, save the model in 'models' directory, and print out the RMSE test result over the testset (which is also being saved in the dataset folder). 

Usage:
```bash
python train.py -fn <str> -dev <int>
```

Input flags: 
   
* -fn: dataset directory name
* -dev: phase devation within the range +-'dev' in degrees that were used for dataset generation
* -s: file name if want to specify it for saving the model
* -opt: optimizaer type: 'ADAM' or 'SGD' (default='ADAM')
* -act: activation function: 'ReLU', 'LeakyReLU' or 'ELU' (default='ReLU')
* -mod: wandb operation mode, use 'disabled' for debugging (default='online')
* -tensor: flag to activate the tensor version of the phase shifter network deviations
* -tx: flag to activate the transmitter calibartion version


main function:
* Need to set the desired hyperparameters for GNN training:
    * num_epochs: number of epochs to train the model
    * batch_size: minibatch size for the dataset iterator
    * learning_rate: learning rate of the optimizaer
    * betas: values for ADAM optimizer
    * weight_decay: L2 penalty coefficient
    * dropout: dropout probability for the MLP used in the GNN
    * d_h: MLP hidden layer dimension for the nodes hidden representation
    * conv_layers: number of graph convolutional layers in the MPNN module
    * mlp_layers: number of layers used in the MLPs of the GNN

    Note: need to replace '<...>' in varaible 'dirc' with the full path name of the directory which contain this script
    

### 3. test.py
Test a choosen GNN model type based on the given dataset and model name, and print out the RMSE test result over the testset. 

Usage:
```bash
python test.py -fn <str>  -dev <int> -model <str>
```

Input flags: 
   
* -fn: dataset directory name
* -dev: phase devation within the range +-'dev' in degrees that were used for dataset generation
* -model: GNN model .pth file full name 

Note: need to replace '<...>' in varaible 'dirc' with the full path name of the directory which contain this script


### 4. models.py
File that contain the GNN Pytorch-based DGL model 'GraphNeuralNetwork' and the classes that form it.
Moreover, the file contains training loop functions (+ validation) and testing loop functions, which are used by 'train.py' and 'test.py'.


### 5. utils.py
File which contains utilities functions such as DGL graph dataset object and saving functions.

Note: need to replace '<...>' in the the init function input 'path' under class 'GraphDataset'. Use the full path name of the directory which contain this script


```
Remark 1: The main directory should include the mentioned files, and two extra directories named: 'data' and 'models'. 
```
```
Remark 2: wandb package is optional, and was used for visualizing the training phase. 
```

