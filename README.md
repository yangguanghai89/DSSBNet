# DSSBNet
# Introduction

This repository contains the implementation code for paper:
DSSBNet: Semantic Decomposition Based on Domain Balance for Patent relevance Assessment.

# Reuirements
Hardware configuration: Ubuntu 22.04 LTS operating system with Intel core i9-14900k CPU and Nvidia GeForce Rtx 4090D GPU.

```
conda create -n my_env python=3.9
pip install -r requirements.txt
```
# data
Please download our dataset from the following address: <https://huggingface.co/datasets/yangguanghai89/DSSBNet_data>

## 🚀 Model Training Setup

Before training the model, follow these steps:

1.  **📁 Configure Paths**
    Open `tool/utilis.ty` and modify the `get_papere` function to update the paths for your dataset and PLM.

2.  **📂 Create Checkpoint Folder**
    Create a folder named `save` to store training checkpoints.

3.  **▶️ Run Training**
    Execute `train.py` to start the training process.

---

💡 **Pre-trained Checkpoints**

We provide the best checkpoints from our training process. Download them here:  
<https://huggingface.co/yangguanghai89/DSSBNet_checkpoint>

## 🧪 Test

To evaluate the model, load the checkpoint with the lowest validation loss in `test.py`. You can configure this within the `main` function of the file. The `za/` directory is used to store reordered output results.

**Example usage:**
```train(path='save/350_0.05376943923049392_0.33214063856464165.pth', args=args, address='za/')```
In the checkpoint path `save/350_0.05376943923049392_0.33214063856464165.pth`:
- **350**: Training step number
- **0.05376943923049392**: Training set loss
- **0.33214063856464165**: Validation set loss









