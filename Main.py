#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import *
from models.test import test_img
import random
import pdb

def Main(args):
    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    Avg_Power = []
    SNR = []
    acc_store = np.array([])
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):

        if np.mod(iter,5) == 0:
            # testing
            net_glob.eval()
            acc_test, _ = test_img(net_glob, dataset_test, args)
            # SNR.append(signal_power/args.variance)
            acc_store = np.append(acc_store, acc_test.numpy())
            print(acc_store)
            net_glob.train()

        history_dict =  net_glob.state_dict()

        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), history_dict=history_dict)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # update global weights
        #pdb.set_trace()
        
        if args.Aligned == 1: # no misalignments, no noise
            current_dict,signal_power = FedAvg(w_locals, args)
        elif args.Aligned == 2: # channel misalignemnt and noise
            current_dict,signal_power = Naive_LMMSE_Decoding(w_locals, args, Async_flag = 0)
        elif args.Aligned == 3: # time, channel misalignemnts and noise, but only Naive and LMMSE decoding
            current_dict,signal_power = Naive_LMMSE_Decoding(w_locals, args, Async_flag = 1)
        else: # time, channel misalignemnts and noise, MAP decoding
            current_dict,signal_power = ASyn_SPA_MAP(w_locals, args)

        for k in current_dict.keys():
            w_glob[k] = history_dict[k] + current_dict[k]

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        Avg_Power.append(signal_power)


    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    acc_store = np.append(acc_store, acc_test.numpy())
    
    return np.array([acc_store])



if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    EsN0dB = np.arange(2,21,2)

    
    for idx in range(len(EsN0dB)):
        args.EsN0dB = EsN0dB[idx]
        print("Running args.EsN0dB = ", args.EsN0dB)
        outputSeq = Main(args)
        if idx == 0:
            StoreRes = outputSeq
        else:
            StoreRes = np.r_[StoreRes,outputSeq]

        print(StoreRes)

    import pickle
    pickle.dump(StoreRes, open('./Results', 'wb'))