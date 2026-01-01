# GPT-2 Reproduction (PyTorch)

This repository contains a from-scratch implementation of the GPT-2 (124M parameter) language model. The project serves as a technical study of modern Large Language Model (LLM) architecture, training dynamics, and efficiency optimizations.

The implementation is based on the [build-nanogpt](https://github.com/karpathy/build-nanogpt) curriculum by Andrej Karpathy, adapted to demonstrate proficiency in PyTorch, distributed computing, and transformer mechanics.

## Technical Architecture

The model follows the standard GPT-2 decoder-only transformer architecture with specific engineering optimizations:

* **Model Specification:** 12 layers, 12 heads, 768 embedding dimension, and a vocabulary size of 50,257.
* **Attention Mechanism:** Implements `CausalSelfAttention` using PyTorch's `F.scaled_dot_product_attention` (Flash Attention) for memory efficiency.
* **Weight Tying:** Implements weight sharing between the token embedding layer (`wte`) and the final linear head (`lm_head`) to improve parameter efficiency.
* **Initialization:** Custom parameter initialization scaling (std $\times (2 \times n\_layer)^{-0.5}$) for residual projections to stabilize deep network training.

## Training Pipeline

The training loop is engineered for high-performance computing contexts:

* **Distributed Training:** Fully integrated `DistributedDataParallel` (DDP) support for multi-GPU training.
* **Precision:** Mixed-precision training using `bfloat16` via `torch.autocast`.
* **Optimization:**
    * Fused AdamW optimizer for kernel-level efficiency.
    * Cosine learning rate decay schedule with linear warmup.
    * Gradient accumulation to simulate larger batch sizes (0.5M tokens) independent of GPU memory constraints.

## Dataset & Evaluation

* **Data:** The model is configured to train on the **FineWeb-Edu** dataset (10B tokens), streamed via `numpy` memory mapping for efficiency.
* **Validation:** Continuous tracking of validation loss and generation of samples during training.
* **Benchmarks:** Integrated evaluation on the **HellaSwag** reasoning benchmark to monitor downstream task performance.

## Usage

### Dependencies
* PyTorch
* TikToken
* NumPy
* HellaSwag (custom module)

### Training
To launch a standard training run:
```bash
python train_gpt2.py
