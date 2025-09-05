# DSSBNet
# Introduction

This repository contains the implementation code for paper:
DSSBNet: Semantic Decomposition Based on Domain Balance for Patent relevance Assessment.

# Reuirements
Hardware configuration: Ubuntu 22.04 LTS operating system with Intel core i9-14900k CPU, 384GB of RAM, and Nvidia GeForce Rtx 4090D GPU.

```bash
conda create -n my_env python=3.9
pip install -r requirements.txt
```
# data
Please download our dataset from the following address: <https://github.com>

# train
Before training the model, open the utilis.ty file in the tool file to modify the path of the dataset and PLM in the get-papere function, and create a folder named save to store the checkpoints. Then run the train.py file. We have released the best checkpoints during our training process, which you can download from the following address:<https://github.com>
