# GPT-2 Reproduction (PyTorch)

This repository contains a from-scratch implementation of the GPT-2 (124M parameter) language model. The project serves as a technical study of modern Large Language Model (LLM) architecture, training dynamics, and efficiency optimizations.

The implementation is based on the [build-nanogpt](https://github.com/karpathy/build-nanogpt) curriculum by Andrej Karpathy, adapted to demonstrate proficiency in PyTorch, distributed computing, and transformer mechanics.

## Project Structure & Implemented Modules

This project is built iteratively, starting from foundational concepts and moving towards the full GPT-2 architecture. The repository currently includes:

* **`bigram.py`**: A baseline Bigram Language Model. This script establishes a simple probabilistic baseline to compare against more complex architectures.
* **`gpt.py`**: A decoder-only Transformer implementation (nanoGPT) trained on the **Tiny Shakespeare** dataset (`input.txt`). It features a configurable architecture (`n_embd`, `n_head`, `n_layer`) and implements the core mechanisms of self-attention and feed-forward networks.
* **`build_GPT.ipynb`**: An educational Jupyter notebook that documents the step-by-step construction of the GPT model, exploring the "under-the-hood" components that make systems like ChatGPT work.

## Tokenization

Tokenization is at the heart of much of the weirdness and capability of LLMs. This project includes a dedicated exploration (`Tokenization.ipynb`) of this critical step.

Key areas covered include:
* **Why LLMs struggle with simple tasks:** Understanding why issues like spelling words, reversing strings, and simple arithmetic are difficult for models due to tokenization boundaries.
* **Multilingual & Structural Issues:** Analysis of why performance degrades on non-English languages and how specific strings (e.g., `<|endoftext|>`, trailing whitespace) can cause unexpected behavior.
* **Implementation:** Moving beyond character-level encoding to more advanced schemes like Byte-Pair Encoding (BPE).

## Technical Architecture (GPT-2 Target)

The full model follows the standard GPT-2 decoder-only transformer architecture with specific engineering optimizations:

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

### Running the Models
To train the baseline or nanoGPT models:
```bash
python bigram.py
python gpt.py
```
To launch a standard GPT-2 training(coming...) run:
```bash
python train_gpt2.py
```
## Project Status
**Note:** This project is currently ongoing. I am actively developing and refining the codebase. This repository serves as a live demonstration of my current progress and technical skills.
