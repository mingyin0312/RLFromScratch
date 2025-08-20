# RLFromScratch

This repo implements **Group Relative Policy Optimization (GRPO)** and **Direct Preference Optimization (DPO)** from scratch in PyTorch, without relying on off-the-shelf libraries like [TRL](https://github.com/huggingface/trl) or [VERL](https://github.com/volcengine/verl).

- GRPO paper: [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)  
- DPO paper: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)

## Why this repo

To open the black box: we unpack the training details—masking, KL penalties, scheduling, and evaluation—so you can see exactly how these algorithms work in practice.

## Quick results
- **GRPO** on **Llama-3.2-1B-Instruct** (GSM8K): **~10% → ~23%** accuracy in **1 epoch**.  
- **DPO** on **Llama-3.2-1B** using **Tiny-Safe-Pair** ([safe-pair-data](https://huggingface.co/datasets/Mingyin0312/safe-pair-data)): **~50% → ~60%** preference accuracy in **3 epochs**.

Both evaluation pipelines are included.

## Training setup

The scripts default to **multi-GPU** training with **PyTorch DDP**, and can be **easily adapted to a single GPU** by adjusting the launch command and disabling distributed initialization. The evaluation is preformed using a single GPU. 

- **Training:**  
  ```bash
  torchrun --standalone --nproc_per_node=8 dpo/grpo_train_from_scratch.py

- **Training:**  
  ```bash
  torchrun --standalone --nproc_per_node=8 dpo/grpo_evaluation.py


## Algorithm Resources

I’ve written explanations of the two algorithms in the following blogs:

- [DPO](https://mingyin0312.github.io/blog/2025/dpo/) 
- [GRPO](https://mingyin0312.github.io/blog/2025/grpo/)

---