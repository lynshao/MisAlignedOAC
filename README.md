# Federated Edge Learning with Misaligned Over-The-Air Computation

This repository is the official implementation of paper [Federated Edge Learning with Misaligned Over-The-Air Computation] (https://arxiv.org/abs/2102.13604)

> If you find this repository useful, please kindly cite as
> 
> @article{techFL,
> 
> title={Federated Learning with Misaligned over-the-air computation},
> 
> author={Shao, Yulin and Gunduz, Deniz and Liew, Soung Chang},
> 
> journal={IEEE Trans. Wireless Commun.},
> 
> volume={21},
> 
> number={6},
> 
> pages={3951-3964},
> 
> year={2022}
> 
> }

## Requirements

Experiments were conducted on Python 3.8.5. To install requirements:

```setup
pip install -r requirements.txt
```

## Case 1: Aligned Channel, with no AWGN

Please run
```train
python main_fed_snr.py --Aligned 0
```

## Case 2: Misaligned channel, with symbol misalignment, phase misalignment, and AWGN

Please run
```train
python main_fed_snr.py --Aligned 1 --maxDelay 0.9 --phaseOffset 0 --Estimator 1
```
> where maxDelay is in range (0,1); phaseOffset can be 0 (no phase misalignment), 1 (maximum phase offset = pi/2),  2 (maximum phase offset = 3pi/4), 3 (maximum phase offset = pi); Estimator can be 1 (aligned sample estimator), 2 (ML estimator), 3 (SP-ML estiamtor)

## Acknowledgement

K. Liu. Train CIFAR-10 with PyTorch. Available online: https://github.com/kuangliu/pytorch-cifar, MIT license, 2020.

S. Ji. A PyTorch implementation of federated learning. Available online: https://github.com/shaoxiongji/federated-learning, MIT license, 2018.

