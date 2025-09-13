# DSSBNet
# Introduction

This repository contains the implementation code for paper:
DSSBNet: Towards Domain-Balanced Semantic Decomposition for Patent Document Re-ranking.

# Abstract
Recent studies increasingly leverage pre-trained language models (PLMs) or large language models (LLMs) to realize technical semantics embedding for patent document re-ranking. Generally, whether a query and the candidate share the technical domain is a critical factor in patent document reranking. However, current works ignore domain information and result in domain-insensitive technical modeling. Meanwhile, distributed discrepancies in training samples across different technical domains introduce domain bias. To address these challenges, we propose a Domain-Sensitive Sample Balance Network (DSSBNet), realizing semantic decomposition based on domain balance, to evaluate technical relevance. It decomposes technical semantics into domain-sensitive and domain-insensitive parts via domain regression and semantic mutual exclusion, which forces the model to boost domain recognition ability in technical semantic modeling. Specifically, a domain balance module is designed through sample reweighting technique and distributed discrepancy constraint to address domain bias for domain-sensitive technical semantics. We conducted experimental evaluations on public dataset CLEF-IP 2011, and the results demonstrate that semantic decomposition significantly enhances the capability in domain semantic comprehension, while domain balance effectively mitigates the adverse effects caused by sample distributed discrepancies. Experimental results demonstrate the improvements of 2% ∼ 15% in recall, 2% ∼ 9% in MAP, as well as 2%-14% in PRES, compared to the best ranking performance of different baselines. All those prove the robustness and higher ranking performance of DSSBNet, which would promote the engineering applications of PLMs and LLMs in patent document re-ranking.

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









