# Federated Learning with Misaligned Over-The-Air Computation -- source code

This repo is based on https://github.com/shaoxiongji/federated-learning


## Requirements
python>=3.6

pytorch>=0.4

## RUN: perfect channel ->->-> no misalignments, no noise
python main_fed_snr.py --Aligned 0

## RUN: channel misalignment, symbol misalignment, noise
# maxDelay \in (0,1)
# phaseOffset = 0->0; 1->pi/2; 2->3pi/4; 3->pi
# Estimator = 1->aligned_sample,2->ML,3->SP-ML
python main_fed_snr.py --Aligned 1 --maxDelay 0.9 --phaseOffset 0 --Estimator 1



