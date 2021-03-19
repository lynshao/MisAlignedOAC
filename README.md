# Federated Edge Learning with Misaligned Over-The-Air Computation -- source code

This repo is based on https://github.com/shaoxiongji/federated-learning


## Requirements
python>=3.6

pytorch>=0.4

## RUN: perfect channel ->->-> no misalignments, no noise
python main_fed_snr.py --Aligned 0

## RUN: channel misalignment, symbol misalignment, noise
python main_fed_snr.py --Aligned 1 --maxDelay 0.9 --phaseOffset 0 --Estimator 1

=> maxDelay \in (0,1)

=> phaseOffset = 0->0; 1->pi/2; 2->3pi/4; 3->pi

=> Estimator = 1->aligned_sample,2->ML,3->SP-ML


# Cite as

If this repo helps, pls kindly cite our work as

@article{Shao2021,

  title={Federated edge learning with misaligned over-the-air computation},
  
  author={Shao, Yulin and Gunduz, Deniz and Liew, Soung Chang},
  
  journal={arXiv preprint arXiv:2102.13604},
  
  year={2021}
  
}

