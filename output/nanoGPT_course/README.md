# Understanding nanoGPT

nanoGPT provides a streamlined framework for understanding and training GPT models. It encapsulates the core GPT Transformer architecture, handles data preparation and loading, orchestrates the entire training process, and enables text generation from a trained model in a clean, single-file implementation approach.

## Contents

| Notebook | Description |
|----------|-------------|
| [01. Configuring Your GPT Model](notebooks/01_configuring_your_gpt_model.ipynb) | Students will define the essential hyperparameters for a GPT model using a dataclass, understanding how these parameters influence the model's size and capabilities. |
| [02. Building the GPT Transformer Block](notebooks/02_building_the_gpt_transformer_block.ipynb) | Students will construct the core components of the GPT model, including Layer Normalization, Causal Self-Attention, and the MLP, assembling them into a Transformer Block and finally the full GPT model. |
| [03. Preparing and Loading Text Data](notebooks/03_preparing_and_loading_text_data.ipynb) | Students will learn how raw text is tokenized, converted to numerical IDs, saved in an efficient binary format, and then loaded in batches for training, focusing on `get_batch`. |
| [04. Orchestrating the GPT Training Loop](notebooks/04_orchestrating_the_gpt_training_loop.ipynb) | Students will implement the full training loop, including model initialization, optimizer configuration, learning rate scheduling, forward/backward passes, and evaluation, connecting the model and data components. |
| [05. Generating Text with a GPT Model](notebooks/05_generating_text_with_a_gpt_model.ipynb) | Students will learn how a trained GPT model generates new text, understanding the iterative prediction process and common sampling strategies like temperature and top-k filtering. |

## Capstone Project

**mini-nanoGPT**: Students will integrate all learned concepts to build a simplified end-to-end version of nanoGPT. This includes defining a model configuration, constructing the GPT architecture, preparing a small dataset, training the model on that data, and finally generating new text with their trained mini-GPT. This project reinforces the entire lifecycle of a language model.

See `capstone/` for instructions, starter code, and tests.

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook
```