import copy
import numpy as np
import torch
from torch import nn
import pdb
import math 
import scipy.io as io
from functools  import partial
from models.BPDecoding import per_pkt_transmission, BP_Decoding
import time
from matplotlib import pyplot as plt

ToIgnore = ["bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked",
"layer1.0.bn1.running_mean", "layer1.0.bn1.running_var", "layer1.0.bn1.num_batches_tracked",
"layer1.0.bn2.running_mean", "layer1.0.bn2.running_var" , "layer1.0.bn2.num_batches_tracked",
"layer1.0.bn3.running_mean", "layer1.0.bn3.running_var" , "layer1.0.bn3.num_batches_tracked",
"layer1.0.bn4.running_mean", "layer1.0.bn4.running_var" , "layer1.0.bn4.num_batches_tracked",
"layer1.0.bn5.running_mean", "layer1.0.bn5.running_var" , "layer1.0.bn5.num_batches_tracked",
"layer1.1.bn1.running_mean", "layer1.1.bn1.running_var" , "layer1.1.bn1.num_batches_tracked",
"layer1.1.bn2.running_mean", "layer1.1.bn2.running_var" , "layer1.1.bn2.num_batches_tracked",
"layer1.1.bn3.running_mean", "layer1.1.bn3.running_var" , "layer1.1.bn3.num_batches_tracked",
"layer1.2.bn1.running_mean", "layer1.2.bn1.running_var" , "layer1.2.bn1.num_batches_tracked",
"layer1.2.bn2.running_mean", "layer1.2.bn2.running_var" , "layer1.2.bn2.num_batches_tracked",
"layer1.2.bn3.running_mean", "layer1.2.bn3.running_var" , "layer1.2.bn3.num_batches_tracked",
"layer1.3.bn1.running_mean", "layer1.3.bn1.running_var" , "layer1.3.bn1.num_batches_tracked",
"layer1.3.bn2.running_mean", "layer1.3.bn2.running_var" , "layer1.3.bn2.num_batches_tracked",
"layer1.3.bn3.running_mean", "layer1.3.bn3.running_var" , "layer1.3.bn3.num_batches_tracked",
"layer2.0.bn1.running_mean", "layer2.0.bn1.running_var" , "layer2.0.bn1.num_batches_tracked",
"layer2.0.bn2.running_mean", "layer2.0.bn2.running_var" , "layer2.0.bn2.num_batches_tracked",
"layer2.0.bn3.running_mean", "layer2.0.bn3.running_var" , "layer2.0.bn3.num_batches_tracked",
"layer2.0.bn4.running_mean", "layer2.0.bn4.running_var" , "layer2.0.bn4.num_batches_tracked",
"layer2.0.bn5.running_mean", "layer2.0.bn5.running_var" , "layer2.0.bn5.num_batches_tracked",
"layer2.1.bn1.running_mean", "layer2.1.bn1.running_var" , "layer2.1.bn1.num_batches_tracked",
"layer2.1.bn2.running_mean", "layer2.1.bn2.running_var" , "layer2.1.bn2.num_batches_tracked",
"layer2.1.bn3.running_mean", "layer2.1.bn3.running_var" , "layer2.1.bn3.num_batches_tracked",
"layer2.2.bn1.running_mean", "layer2.2.bn1.running_var" , "layer2.2.bn1.num_batches_tracked",
"layer2.2.bn2.running_mean", "layer2.2.bn2.running_var" , "layer2.2.bn2.num_batches_tracked",
"layer2.2.bn3.running_mean", "layer2.2.bn3.running_var" , "layer2.2.bn3.num_batches_tracked",
"layer2.3.bn1.running_mean", "layer2.3.bn1.running_var" , "layer2.3.bn1.num_batches_tracked",
"layer2.3.bn2.running_mean", "layer2.3.bn2.running_var" , "layer2.3.bn2.num_batches_tracked",
"layer2.3.bn3.running_mean", "layer2.3.bn3.running_var" , "layer2.3.bn3.num_batches_tracked",
"layer2.4.bn1.running_mean", "layer2.4.bn1.running_var" , "layer2.4.bn1.num_batches_tracked",
"layer2.4.bn2.running_mean", "layer2.4.bn2.running_var" , "layer2.4.bn2.num_batches_tracked",
"layer2.4.bn3.running_mean", "layer2.4.bn3.running_var" , "layer2.4.bn3.num_batches_tracked",
"layer2.5.bn1.running_mean", "layer2.5.bn1.running_var" , "layer2.5.bn1.num_batches_tracked",
"layer2.5.bn2.running_mean", "layer2.5.bn2.running_var" , "layer2.5.bn2.num_batches_tracked",
"layer2.5.bn3.running_mean", "layer2.5.bn3.running_var" , "layer2.5.bn3.num_batches_tracked",
"layer2.6.bn1.running_mean", "layer2.6.bn1.running_var" , "layer2.6.bn1.num_batches_tracked",
"layer2.6.bn2.running_mean", "layer2.6.bn2.running_var" , "layer2.6.bn2.num_batches_tracked",
"layer2.6.bn3.running_mean", "layer2.6.bn3.running_var" , "layer2.6.bn3.num_batches_tracked",
"layer2.7.bn1.running_mean", "layer2.7.bn1.running_var" , "layer2.7.bn1.num_batches_tracked",
"layer2.7.bn2.running_mean", "layer2.7.bn2.running_var" , "layer2.7.bn2.num_batches_tracked",
"layer2.7.bn3.running_mean", "layer2.7.bn3.running_var" , "layer2.7.bn3.num_batches_tracked",
"layer3.0.bn1.running_mean", "layer3.0.bn1.running_var" , "layer3.0.bn1.num_batches_tracked",
"layer3.0.bn2.running_mean", "layer3.0.bn2.running_var" , "layer3.0.bn2.num_batches_tracked",
"layer3.0.bn3.running_mean", "layer3.0.bn3.running_var" , "layer3.0.bn3.num_batches_tracked",
"layer3.0.bn4.running_mean", "layer3.0.bn4.running_var" , "layer3.0.bn4.num_batches_tracked",
"layer3.0.bn5.running_mean", "layer3.0.bn5.running_var" , "layer3.0.bn5.num_batches_tracked",
"layer3.1.bn1.running_mean", "layer3.1.bn1.running_var" , "layer3.1.bn1.num_batches_tracked",
"layer3.1.bn2.running_mean", "layer3.1.bn2.running_var" , "layer3.1.bn2.num_batches_tracked",
"layer3.1.bn3.running_mean", "layer3.1.bn3.running_var" , "layer3.1.bn3.num_batches_tracked",
"layer3.2.bn1.running_mean", "layer3.2.bn1.running_var" , "layer3.2.bn1.num_batches_tracked",
"layer3.2.bn2.running_mean", "layer3.2.bn2.running_var" , "layer3.2.bn2.num_batches_tracked",
"layer3.2.bn3.running_mean", "layer3.2.bn3.running_var" , "layer3.2.bn3.num_batches_tracked",
"layer3.3.bn1.running_mean", "layer3.3.bn1.running_var" , "layer3.3.bn1.num_batches_tracked",
"layer3.3.bn2.running_mean", "layer3.3.bn2.running_var" , "layer3.3.bn2.num_batches_tracked",
"layer3.3.bn3.running_mean", "layer3.3.bn3.running_var" , "layer3.3.bn3.num_batches_tracked",
"bn2.running_mean", "bn2.running_var", "bn2.num_batches_tracked"]

def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)


# When d_M \to 1, we do not need to do message passing.
# Instead, the posterior distribution can be computed directly from the product of f_t and f_b
def per_pkt_transmission_Simple(args, MM, TransmittedSymbols):
    # Pass the channel and generate samples at the receiver
    # the maximum time offset is fixed to 0.99

    # # # Generate the channel: phaseOffset = 0->0; 1->2pi/4; 2->2pi/2; 3->2pi
    if args.phaseOffset == 0:
        hh = np.ones([MM,1])
    elif args.phaseOffset == 1:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/4)
    elif args.phaseOffset == 2:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/2)
    else:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi)

    # complex pass the complex channel
    for idx in range(MM):
        TransmittedSymbols[idx,:] = TransmittedSymbols[idx,:] * hh[idx][0]

    # compute the received signal power and add noise
    LL = len(TransmittedSymbols[0])
    SignalPart = np.sum(TransmittedSymbols,0)

    SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    # SigPower = np.max(np.power(np.abs(SignalPart),2))
    EsN0 = np.power(10, args.EsN0dB/10.0)
    noisePower = SigPower/EsN0 
    
    RecevidSignal = SignalPart + (np.random.normal(0,1,LL)+1j*np.random.normal(0,1,LL)) * np.sqrt(noisePower/2)

    # ################################################### ML estimation
    # ---------------------------------------------------- Prepare the sample info
    beta1 = np.c_[np.real(hh),np.imag(hh)]
    beta2 = np.c_[-np.imag(hh),np.real(hh)]
    Obser_Lamb1 = np.c_[np.matmul(beta1,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta2))]
    Obser_Lamb2 = np.c_[np.matmul(beta2,np.transpose(beta1)),np.matmul(beta1,np.transpose(beta1))]
    Obser_Lamb = np.r_[Obser_Lamb1,Obser_Lamb2]
    etaMat = np.matmul(np.r_[beta1,beta2],np.r_[np.real([RecevidSignal]),np.imag([RecevidSignal])])

    # ---------------------------------------------------- Estimation
    Sum_mu = np.zeros([2,LL])
    Res_Lamb = Obser_Lamb / (noisePower/2)
    Res_Sigma = np.linalg.pinv(Res_Lamb)

    Res_Eta = etaMat / (noisePower/2)
    Res_mu = np.matmul(Res_Sigma, Res_Eta)

    # compute (mu,Sigma) for the sum
    output = np.append(np.sum(Res_mu[0:MM,:],0), np.sum(Res_mu[MM:,:],0))

    # average weights
    output = output/MM

    return output



def plot_pdf(samples):
    plt.hist(samples, bins = 'auto', density = 0)
    plt.show()