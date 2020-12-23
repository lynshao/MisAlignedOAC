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
from functools import partial
import matlab.engine
import time


def flatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)

##### ================================================ UL FedAvg (no noise, no misalignments)
##### ================================================
def FedAvg(w, args):
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        # get the receivd signal
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]

        w_avg[k] = torch.div(w_avg[k], len(w))

    return w_avg, 0


##### ================================================ Naive and LMMSE Estimators
##### ================================================ Synchronous AirComp / Asynchrnous AirComp (EsN0 Penalty)
##### ================================================

def Naive_LMMSE_Decoding(w, args, Async_flag = 0):
    MM = len(w) # number of devices

    # # # Generate the channel: phaseOffset = 0->0; 1->2pi/4; 2->2pi/2; 3->2pi
    if args.phaseOffset == 0:
        hh = np.ones([MM,1])
    elif args.phaseOffset == 1:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/4)
    elif args.phaseOffset == 2:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/2)
    else:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi)

    # Generate the symbols to be transmitted from the M devices: TransmittedSymbols_real (real); TransmittedSymbols (complex without channel)
    StoreRecover = np.array([])
    for idx in np.arange(MM):
        wEach = w[idx]
        eachWeightNumpy = np.array([])
        for k in wEach.keys():
            temp = wEach[k].cpu().numpy()
            temp, unflatten = flatten(temp)
            if idx == 0:
                StoreRecover = np.append(StoreRecover,unflatten)
            eachWeightNumpy = np.append(eachWeightNumpy, temp)
    
        if idx == 0:
            TransmittedSymbols_real = np.array([eachWeightNumpy]) # for computing the sample mean/variance
            complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.array([complexSymbols])
        else:
            TransmittedSymbols_real = np.r_[TransmittedSymbols_real, np.array([eachWeightNumpy])] # for computing the sample mean/variance
            complexSymbols =  eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]
    
    LL = len(TransmittedSymbols[0])

    # complex pass the complex channel
    # construct the real sequence to compute the samle mean/variance (must be computed after scaling)
    for idx in range(MM):
        TransmittedSymbols[idx,:] = TransmittedSymbols[idx,:] * hh[idx][0]

    # scaling
    TransAmplitude = 10
    TransmittedSymbols_real = TransmittedSymbols_real * TransAmplitude
    TransmittedSymbols = TransmittedSymbols * TransAmplitude

    # compute the sample mean and sample variance
    SamplesMeans = np.mean(TransmittedSymbols_real,1)
    SampleVars = np.var(TransmittedSymbols_real,1)

    # compute the received signal power and add noise
    SignalPart = np.sum(TransmittedSymbols,0)

    SigPower = np.max(np.power(np.abs(SignalPart),2))
    EsN0 = np.power(10, args.EsN0dB/10.0)

    if Async_flag == 0: # synchronous
        noisePower = SigPower/EsN0
    else:
        noisePower = SigPower/EsN0/(1-args.maxDelay)
    
    RecevidSignal = SignalPart + (np.random.normal(0,1,LL)+1j*np.random.normal(0,1,LL)) * np.sqrt(noisePower/2)


    if args.Estimator == 1: # naive
        output = np.append(np.real(RecevidSignal), np.imag(RecevidSignal))
    elif args.Estimator == 2: # LMMSE
        # compute the LMMSE coefficient cc and dd (complex)
        # compute the sample mean and sample variance
        SamplesMeans = np.mean(TransmittedSymbols,1)
        SampleVars = np.var(TransmittedSymbols,1)
        SampleDD = SampleVars - np.power(np.abs(SamplesMeans),2)
        DD = np.diag(SampleDD)
        # ---------------------------------- dd = 0
        # V = np.zeros([MM,MM]) + (1j*0)
        # for i1 in range(MM):
        #     for i2 in range(MM):
        #         if i1 == i2:
        #             V[i1,i2] = SampleVars[i1]
        #         else:
        #             V[i1,i2] = np.conj(SamplesMeans[i1]) * SamplesMeans[i2]
        # cc = np.matmul(np.matmul(hh.conj().T, V), np.ones([MM,1]))[0] / (np.matmul(np.matmul(hh.conj().T, V), hh) + noisePower)[0]
        # RecevidSignal = cc * RecevidSignal

        # ---------------------------------- dd ~= 0
        cc = np.matmul(np.matmul(hh.conj().T, DD), np.ones([MM,1]))[0] / (np.matmul(np.matmul(hh.conj().T, DD), hh) + noisePower)[0]
        dd = np.matmul((np.ones([MM,1]) - cc * hh).T, np.array([SamplesMeans]).T)[0]
        RecevidSignal = cc * RecevidSignal + dd

        output = np.append(np.real(RecevidSignal), np.imag(RecevidSignal))
    else:
        print("Error Input: args.Estimator")
    
    # average weights
    output = output/TransAmplitude/MM

    ## Transform the numpy weights to torch in cuda()
    w_avg = copy.deepcopy(w[0])

    startIndex = 0
    idx = 0
    for k in w_avg.keys():
        lenLayer = w_avg[k].numel()
        # get data
        ParamsLayer = output[startIndex:(startIndex+lenLayer)]
        # reshape
        ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
        # convert to torch in cuda()
        w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()

        startIndex += lenLayer
        idx += 1

    
    return w_avg, 0

##### ================================================ SPA-MAP Estimator
##### ================================================ Asynchronous AirComp 
##### ================================================

def ASyn_SPA_MAP(w, args):
    MM = len(w) # number of devices

    # Generate the symbols to be transmitted from the M devices: TransmittedSymbols_real (real); TransmittedSymbols (complex without channel)
    StoreRecover = np.array([])
    for idx in np.arange(MM):
        wEach = w[idx]
        eachWeightNumpy = np.array([])
        for k in wEach.keys():
            temp = wEach[k].cpu().numpy()
            temp, unflatten = flatten(temp)
            if idx == 0:
                StoreRecover = np.append(StoreRecover,unflatten)
            eachWeightNumpy = np.append(eachWeightNumpy, temp)
    
        if idx == 0:
            complexSymbols = eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.array([complexSymbols])
        else:
            complexSymbols =  eachWeightNumpy[0:int(len(eachWeightNumpy)/2)] + 1j * eachWeightNumpy[int(len(eachWeightNumpy)/2):]
            TransmittedSymbols = np.r_[TransmittedSymbols, np.array([complexSymbols])]
    
    LL = len(TransmittedSymbols[0])

    eng = matlab.engine.start_matlab()
    onePkt = TransmittedSymbols
    ReceivedPkt = per_pkt_transmission(args, MM, onePkt, eng)
    eng.quit()

    ## ========================================== Transform the numpy weights to torch in cuda()
    ## ========================================================================================= 
    startIndex = 0
    idx = 0

    ## Transform the numpy weights to torch in cuda()
    w_avg = copy.deepcopy(w[0])

    for k in w_avg.keys():
        lenLayer = w_avg[k].numel()
        # get data
        ParamsLayer = ReceivedPkt[startIndex:(startIndex+lenLayer)]
        # reshape
        ParamsLayer_reshaped = StoreRecover[idx](ParamsLayer)
        # convert to torch in cuda()
        w_avg[k] = torch.from_numpy(ParamsLayer_reshaped).cuda()

        startIndex += lenLayer
        idx += 1

    
    return w_avg, 0

def per_pkt_transmission(args, MM, TransmittedSymbols, eng):
    # Pass the channel and generate samples at the receiver
    taus = np.sort(np.random.uniform(0,args.maxDelay,(1,MM-1)))[0]
    taus[-1] = args.maxDelay
    dd = np.zeros(MM)
    for idx in np.arange(MM):
        if idx == 0:
            dd[idx] = taus[0]
        elif idx == MM-1:
            dd[idx] = 1 - taus[-1]
        else:
            dd[idx] = taus[idx] - taus[idx-1]

    # # # Generate the channel: phaseOffset = 0->0; 1->2pi/4; 2->2pi/2; 3->2pi
    if args.phaseOffset == 0:
        hh = np.ones([MM,1])
    elif args.phaseOffset == 1:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/4)
    elif args.phaseOffset == 2:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi/2)
    else:
        hh = np.exp(1j*np.random.uniform(0,1,(MM,1)) * 2* np.pi)

    # scaling
    TransAmplitude = 10
    TransmittedSymbols = TransmittedSymbols * TransAmplitude

    # complex pass the complex channel
    # construct the real sequence to compute the samle mean/variance (must be computed after scaling)
    for idx in range(MM):
        if idx == 0:
            TransmittedSymbols_real = np.array([np.append(np.real(TransmittedSymbols[idx,:]),np.imag(TransmittedSymbols[idx,:]))])
        else:
            TransmittedSymbols_real = np.r_[TransmittedSymbols_real, np.array([np.append(np.real(TransmittedSymbols[idx,:]),np.imag(TransmittedSymbols[idx,:]))])]

        TransmittedSymbols[idx,:] = TransmittedSymbols[idx,:] * hh[idx][0]

    # compute the sample mean and sample variance
    SamplesMeans = np.mean(TransmittedSymbols_real,1)
    SampleVars = np.var(TransmittedSymbols_real,1)

    # compute the received signal power and add noise
    LL = len(TransmittedSymbols[0])
    SignalPart = np.sum(TransmittedSymbols,0)

    # SigPower = np.sum(np.power(np.abs(SignalPart),2))/LL
    SigPower = np.max(np.power(np.abs(SignalPart),2))
    EsN0 = np.power(10, args.EsN0dB/10.0)
    noisePower = SigPower/EsN0

    # Oversample the received signal
    RepeatedSymbols = np.repeat(TransmittedSymbols, MM, axis = 1)
    for idx in np.arange(MM):
        extended = np.array([np.r_[np.zeros(idx), RepeatedSymbols[idx], np.zeros(MM-idx-1)]])
        if idx == 0:
            samples = extended
        else:
            samples = np.r_[samples, extended]
    
    samples = np.sum(samples, axis=0)
    
    # generate noise
    for idx in np.arange(MM):
        noise = np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)+1j*np.random.normal(loc=0, scale=np.sqrt(noisePower/2/dd[idx]), size=LL+1)
        if idx == 0:
            AWGNnoise = np.array([noise])
        else:
            AWGNnoise = np.r_[AWGNnoise, np.array([noise])]

    AWGNnoise = np.reshape(AWGNnoise, (1,MM*(LL+1)), 'F')
    samples = samples + AWGNnoise[0][0:-1]
    Transmitted_real = np.append(np.real(samples),np.imag(samples))

    # ================================================= IMPORT TO MATLAB TO PROCESS
    # startTime = time.time()
    para_seq1 = np.r_[dd,np.real(hh)[:,0], np.imag(hh)[:,0]]
    para_seq2 = np.r_[SamplesMeans, SampleVars, np.array([noisePower]), np.array([LL])]
    ans = eng.BPDecoding(matlab.double(Transmitted_real.tolist()),matlab.double(para_seq1.tolist()),matlab.double(para_seq2.tolist()))
    output = np.array(ans._data)

    # average weights
    output = output/TransAmplitude/MM

    return output