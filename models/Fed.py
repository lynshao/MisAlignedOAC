#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import numpy as np
import torch
from torch import nn
import pdb
import math 
import scipy.io as io
from functools  import partial
from models.BPDecoding import per_pkt_transmission, BP_Decoding
from utils.Ignore import ToIgnore, flatten, plot_pdf
import time
from matplotlib import pyplot as plt
from multiprocessing import Process, Queue, cpu_count, Pool
import os, random

# ================================================ perfect channel ->->-> no misalignments, no noise
def FedAvg(w, args, flag = 0):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # get the receivd signal
        if (flag == 1) and (k not in ToIgnore):
            continue

        for i in range(1, len(w)):
            w_avg[k] += w[i][k]

        # weight_coLector.append((w_avg[k]-noise).cpu().numpy())

        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

# ================================================ Asynchronous AirComp
def FedAvg_ML(w, args):
    M = len(w) # number of devices

    # ----------------------------------- Extract the symbols (weight updates) from each devices as a numpy array (complex sequence)
    StoreRecover = np.array([]) # record how to transform the numpy arracy back to dictionary
    for m in np.arange(M):
        # extract symbols from one device (layer by layer)
        wEach = w[m]
        eachWeightNumpy = np.array([])
        for k in wEach.keys():
            # The batch normalization layers should be ignroed as they are not weights (can be transmitted reliably to the PS in practice)
            if k in ToIgnore:
                continue
            temp = wEach[k].cpu().numpy()
            temp, unflatten = flatten(temp)
            if m == 0:
                StoreRecover = np.append(StoreRecover,unflatten)
            eachWeightNumpy = np.append(eachWeightNumpy, temp)
    
        # stack the symbols from different devices ->-> numpy array of shape M * L
        if m == 0:
            complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.array([complexSymbols])
        else:
            complexSymbols =  eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]
    
    # number of complex symbols from each device
    L = len(TransmittedSymbols[0]) # 631927
    
    # ---------------------------------------------------------------------------------- pkt by pkt transmission
    # add 1 all-zero column => 631927 + 1 = 631928 (11 * 4 * 43 * 167 * 2) = 57448 * 11 or 28724 * 22
    TransmittedSymbols = np.c_[TransmittedSymbols,np.zeros([M,1])]  # 4 * 631928

    # -------------------------------------------------------- long pkts
    if args.short == 0:
        numPkt = 44
        lenPkt = int((L+1)/numPkt)
        pool = Pool(processes=numPkt) # creat 11 processes
        results = []
        for idx in range(numPkt):
            # transmissted complex symbols in one pkt
            onePkt = TransmittedSymbols[:,(idx*lenPkt):((idx+1)*lenPkt)]
            # received complex symbols in each pkt (after estmimation and averaging)
            results.append(pool.apply_async(per_pkt_transmission, (args, M, onePkt, )))

        pool.close() # close the pool, no more processes added
        pool.join() # waiting for all processes done
     
        for idx in range(len(results)):
            try:
                output = results[idx].get()
            except:
                pdb.set_trace()
            if idx == 0:
                ReceivedComplexPkt = output
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, output)
    else:
        # -------------------------------------------------------- short pkts
        numPkt = 946 # 22 * 43
        lenPkt = int((L+1)/numPkt)
        # multi-processing
        # numCPU = cpu_count()
        numCPU = 22
        for loop in range(43):
            pktBatch = TransmittedSymbols[:,(loop*lenPkt*numCPU):((loop+1)*lenPkt*numCPU)]
            pool = Pool(processes=numCPU) # creat 11 processes
            results = []
            for idx in range(numCPU):
                # transmissted complex symbols in one pkt
                onePkt = pktBatch[:,(idx*lenPkt):((idx+1)*lenPkt)]
                # received complex symbols in each pkt (after estmimation and averaging)
                results.append(pool.apply_async(per_pkt_transmission, (args, M, onePkt, )))

            pool.close() # close the pool, no more processes added
            pool.join() # waiting for all processes done
         
            for idx in range(len(results)):
                try:
                    output = results[idx].get()
                except:
                    pdb.set_trace()
                if idx == 0:
                    ReceivedBatch = output
                else:
                    ReceivedBatch = np.append(ReceivedBatch, output)

            if loop == 0:
                ReceivedComplexPkt = ReceivedBatch
            else:
                ReceivedComplexPkt = np.append(ReceivedComplexPkt, ReceivedBatch)

    # Restore the real weights ->->-> numpy array
    ReceivedPkt = np.append(np.real(ReceivedComplexPkt[:-1]), np.imag(ReceivedComplexPkt[:-1])) # the last element (0) must be deleted

    ## ========================================================================================= 
    ## ========================================================================================= 
    ## ========================================== Reconstruct the dictionary from the numpy array
    # run the federated averaging first (to tackle the batched normalization layer)
    w_avg = FedAvg(w, args, 1)

    startIndex = 0
    idx = 0
    for k in w_avg.keys():
        # only update the non-batched-normalization-layers in w_avg
        if k not in ToIgnore:
            lenLayer = w_avg[k].numel()
            # get data
            ParamsLayer = ReceivedPkt[startIndex:(startIndex+lenLayer)]
            # reshape
            ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
            # convert to torch in cuda()
            w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()

            startIndex += lenLayer
            idx += 1

    
    return w_avg









