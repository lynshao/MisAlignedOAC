#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import *
from models.shufflenetv2 import *
from models.Fed import *
from models.test import test_img
import random, time, pdb, pickle

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def Main(args):
    setup_seed(args.seed)
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
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=transform_train)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=transform_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users = cifar_noniid(dataset_train, args.num_users)
    else:
        exit('Error: unrecognized dataset')

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        # net_glob = CNNCifar(args=args).to(args.device)
        net_glob = ShuffleNetV2(1).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    else:
        exit('Error: unrecognized model')
    net_glob.train()

    print("============================== Federated Learning ... ...")

    # training
    loss_train = []
    acc_store = np.array([])
    w_glob = net_glob.state_dict() # initial global weights
    for iter in range(args.epochs):
        # record the running time of an iteration
        startTime = time.time()

        # Every 5 itertaions, evaluate the learning performance in terms of "test accuracy"
        if np.mod(iter,5) == 0:
            net_glob.eval()
            acc_test, _ = test_img(net_glob, dataset_test, args)
            acc_store = np.append(acc_store, acc_test.numpy())
            print("Test accuracies every 5 itertaions =", acc_store)
            net_glob.train()

        # ----------------------------------------------------------------------- Preparation
        # global model at the beginning of the iteration
        history_dict =  net_glob.state_dict()

        # random choose M devices out of the args.num_users devices
        M = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), M, replace=False)
        args.lr = args.lr * 0.95 # learning rate adjustment

        # ----------------------------------------------------------------------- Local Training
        w_locals = [] # store the local "updates" (the difference) of M devices
        loss_locals = [] # store the local training loss
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device), history_dict=history_dict)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        # ----------------------------------------------------------------------- Federated Averaging
        if args.Aligned == 0: # perfect channel ->->-> no misalignments, no noise
            current_dict = FedAvg(w_locals, args)
        elif args.Aligned == 1: # channel misalignment, symbol misalignment, noise
            current_dict = FedAvg_ML(w_locals, args)
        else:
            exit("unknown args.Aligned")

        # ----------------------------------------------------------------------- Reconstruct the new model at the PS
        for k in current_dict.keys():
            w_glob[k] = history_dict[k] + current_dict[k]
        # load new model
        net_glob.load_state_dict(w_glob)

        # print training loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}, Time Cosumed {:.3f}'.format(iter, loss_avg, time.time()-startTime))
        loss_train.append(loss_avg)


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

    EsN0dB = np.arange(30,41,5)

    StoreRes = []
    for idx in range(len(EsN0dB)):
        args.EsN0dB = EsN0dB[idx]
        args.lr = 0.1
        print("Running args.EsN0dB = ", args.EsN0dB)
        outputSeq = Main(args)
        StoreRes.append(list(outputSeq))        

        # print and store the results
        print(StoreRes)
        pickle.dump(StoreRes, open('./Results', 'wb'))
