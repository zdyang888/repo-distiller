# Understanding nanoGPT

nanoGPT provides a clean, self-contained implementation for building and training GPT-style large language models. Its core is the `GPT` model, a standard Transformer decoder architecture comprising token embeddings, positional embeddings, a stack of `Block` modules (each with `CausalSelfAttention` and an `MLP`), and a final language modeling head. The `train.py` script manages the entire training lifecycle, from loading model configurations and efficiently prepared data to executing forward/backward passes and logging metrics.

## Contents

| Notebook | Description |
|----------|-------------|
| [01. Defining GPT Model Hyperparameters with GPTConfig](notebooks/01_defining_gpt_model_hyperparameters_with_.ipynb) | Students will define a dataclass to manage all model hyperparameters, understanding how these values dictate the model's size and capabilities. |
| [02. Implementing Causal Self-Attention](notebooks/02_implementing_causal_self_attention.ipynb) | Students will build the `CausalSelfAttention` mechanism, a core component for sequence modeling, learning how to mask future tokens to maintain causality. |
| [03. Constructing the Transformer Block](notebooks/03_constructing_the_transformer_block.ipynb) | Students will integrate the Causal Self-Attention with a Multi-Layer Perceptron (MLP) and apply Layer Normalization and residual connections to form a complete Transformer Block. |
| [04. Assembling the Full GPT Model Architecture](notebooks/04_assembling_the_full_gpt_model_architectu.ipynb) | Students will combine token embeddings, positional embeddings, and a stack of Transformer Blocks to construct the complete `GPT` model, including its language modeling head. |
| [05. Preparing Text Data for GPT Training](notebooks/05_preparing_text_data_for_gpt_training.ipynb) | Students will learn how to process raw text, tokenize it, and convert it into numerical sequences suitable for efficient loading and training of a GPT model. |
| [06. Implementing the GPT Training Loop](notebooks/06_implementing_the_gpt_training_loop.ipynb) | Students will implement the core training loop responsible for initializing the model and optimizer, loading data batches, performing forward/backward passes, and updating model weights. |

## Capstone Project

**mini-nanoGPT**: Students will build a simplified, end-to-end version of nanoGPT, integrating all previously learned components into a functional language model. This project emphasizes how data preparation, model architecture, and training loops come together to create a minimal yet complete system, highlighting the core principles behind modern LLMs.

See `capstone/` for instructions, starter code, and tests.

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook
```