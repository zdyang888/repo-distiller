# Curriculum: Understanding nanoGPT

Source: https://github.com/karpathy/nanoGPT

## Overview

nanoGPT provides a streamlined framework for understanding and training GPT models. It encapsulates the core GPT Transformer architecture, handles data preparation and loading, orchestrates the entire training process, and enables text generation from a trained model in a clean, single-file implementation approach.

## Notebooks

### 01. Configuring Your GPT Model

Students will define the essential hyperparameters for a GPT model using a dataclass, understanding how these parameters influence the model's size and capabilities.

**Learning objectives:**
- By the end, students will be able to define and instantiate a GPTConfig object.
- Students will understand how hyperparameters like n_layer, n_head, and n_embd relate to model complexity and resource usage.

**Exercise:** Students will implement a simple `ModelConfig` dataclass from scratch to define model dimensions (num_layers, num_heads, embedding_dim, block_size).

### 02. Building the GPT Transformer Block

Students will construct the core components of the GPT model, including Layer Normalization, Causal Self-Attention, and the MLP, assembling them into a Transformer Block and finally the full GPT model.

**Learning objectives:**
- By the end, students will be able to implement the Causal Self-Attention mechanism.
- Students will understand how Transformer Blocks are composed and stacked to form the GPT architecture.
- Students will connect GPTConfig parameters to the instantiated architecture.

**Exercise:** Students will implement a `MiniAttention` layer and an `EncoderBlock` class, then integrate them into a basic `MiniGPT` model using the `ModelConfig` from the previous exercise.

### 03. Preparing and Loading Text Data

Students will learn how raw text is tokenized, converted to numerical IDs, saved in an efficient binary format, and then loaded in batches for training, focusing on `get_batch`.

**Learning objectives:**
- By the end, students will be able to implement basic text tokenization and numericalization.
- Students will understand how to load and prepare batches of data efficiently from binary files for model training.

**Exercise:** Students will implement a `SimpleTextDataset` class that tokenizes a given text and provides a `get_batch` method to yield input-target pairs.

### 04. Orchestrating the GPT Training Loop

Students will implement the full training loop, including model initialization, optimizer configuration, learning rate scheduling, forward/backward passes, and evaluation, connecting the model and data components.

**Learning objectives:**
- By the end, students will be able to implement a complete training loop for a language model.
- Students will understand the role of optimizers, learning rate schedulers, and evaluation metrics in model training.
- Students will learn how to save and load model checkpoints.

**Exercise:** Students will implement a `MiniTrainer` class that takes a model and data loader, then orchestrates the training, evaluation, and checkpointing process.

### 05. Generating Text with a GPT Model

Students will learn how a trained GPT model generates new text, understanding the iterative prediction process and common sampling strategies like temperature and top-k filtering.

**Learning objectives:**
- By the end, students will be able to use a trained GPT model to generate coherent text.
- Students will understand the mechanisms of autoregressive text generation and token sampling techniques.
- Students will be able to implement `temperature` and `top_k` sampling.

**Exercise:** Students will extend their `MiniGPT` model with a `generate` method that takes a prompt and generates new tokens autoregressively, incorporating temperature-based sampling.

## Capstone Project

**mini-nanoGPT**

Students will integrate all learned concepts to build a simplified end-to-end version of nanoGPT. This includes defining a model configuration, constructing the GPT architecture, preparing a small dataset, training the model on that data, and finally generating new text with their trained mini-GPT. This project reinforces the entire lifecycle of a language model.

---

<!-- Edit the JSON below to skip notebooks (set skip: true) -->

```json
{
  "title": "Understanding nanoGPT",
  "mental_model": "nanoGPT provides a streamlined framework for understanding and training GPT models. It encapsulates the core GPT Transformer architecture, handles data preparation and loading, orchestrates the entire training process, and enables text generation from a trained model in a clean, single-file implementation approach.",
  "concepts": [
    {
      "name": "GPT Configuration",
      "description": "A dataclass that defines all hyperparameters necessary to construct a specific GPT model instance, such as the number of layers, heads, embedding dimensions, context size, and dropout rates.",
      "complexity": "basic"
    },
    {
      "name": "GPT Architecture",
      "description": "The neural network structure of the GPT model, composed of modular building blocks like Layer Normalization, Causal Self-Attention, and Multi-Layer Perceptrons (MLP), stacked into Transformer Blocks.",
      "complexity": "intermediate"
    },
    {
      "name": "Data Preparation & Loading",
      "description": "The process of converting raw text into numerical token IDs, storing them in an efficient binary format, and loading these token sequences into batches for training.",
      "complexity": "basic"
    },
    {
      "name": "Training Orchestration",
      "description": "The comprehensive workflow for training a GPT model, encompassing model initialization, optimization (AdamW), learning rate scheduling, evaluation, checkpoint saving, and support for distributed training.",
      "complexity": "intermediate"
    },
    {
      "name": "Text Generation/Sampling",
      "description": "The inference process where a trained GPT model generates new text by feeding an initial prompt and iteratively predicting the next token using strategies like temperature sampling and top-k filtering.",
      "complexity": "basic"
    }
  ],
  "notebooks": [
    {
      "id": "01",
      "title": "Configuring Your GPT Model",
      "concept": "GPT Configuration",
      "description": "Students will define the essential hyperparameters for a GPT model using a dataclass, understanding how these parameters influence the model's size and capabilities.",
      "prerequisites": [],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "GPTConfig"
      ],
      "learning_objectives": [
        "By the end, students will be able to define and instantiate a GPTConfig object.",
        "Students will understand how hyperparameters like n_layer, n_head, and n_embd relate to model complexity and resource usage."
      ],
      "exercise_description": "Students will implement a simple `ModelConfig` dataclass from scratch to define model dimensions (num_layers, num_heads, embedding_dim, block_size).",
      "visualization_idea": "A diagram showing how each configuration parameter maps to a conceptual part of the GPT model (e.g., n_layer = number of stacked blocks)."
    },
    {
      "id": "02",
      "title": "Building the GPT Transformer Block",
      "concept": "GPT Architecture",
      "description": "Students will construct the core components of the GPT model, including Layer Normalization, Causal Self-Attention, and the MLP, assembling them into a Transformer Block and finally the full GPT model.",
      "prerequisites": [
        "GPT Configuration"
      ],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "LayerNorm",
        "CausalSelfAttention",
        "MLP",
        "Block",
        "GPT"
      ],
      "learning_objectives": [
        "By the end, students will be able to implement the Causal Self-Attention mechanism.",
        "Students will understand how Transformer Blocks are composed and stacked to form the GPT architecture.",
        "Students will connect GPTConfig parameters to the instantiated architecture."
      ],
      "exercise_description": "Students will implement a `MiniAttention` layer and an `EncoderBlock` class, then integrate them into a basic `MiniGPT` model using the `ModelConfig` from the previous exercise.",
      "visualization_idea": "An animated flow illustrating data passing through LayerNorm, Causal Self-Attention, and MLP within a single Transformer block."
    },
    {
      "id": "03",
      "title": "Preparing and Loading Text Data",
      "concept": "Data Preparation & Loading",
      "description": "Students will learn how raw text is tokenized, converted to numerical IDs, saved in an efficient binary format, and then loaded in batches for training, focusing on `get_batch`.",
      "prerequisites": [],
      "key_source_files": [
        "data/shakespeare_char/prepare.py",
        "train.py"
      ],
      "key_symbols": [
        "get_batch"
      ],
      "learning_objectives": [
        "By the end, students will be able to implement basic text tokenization and numericalization.",
        "Students will understand how to load and prepare batches of data efficiently from binary files for model training."
      ],
      "exercise_description": "Students will implement a `SimpleTextDataset` class that tokenizes a given text and provides a `get_batch` method to yield input-target pairs.",
      "visualization_idea": "An animation showing raw text transforming into token IDs, then into `x` and `y` tensors for a batch, highlighting the `block_size` concept."
    },
    {
      "id": "04",
      "title": "Orchestrating the GPT Training Loop",
      "concept": "Training Orchestration",
      "description": "Students will implement the full training loop, including model initialization, optimizer configuration, learning rate scheduling, forward/backward passes, and evaluation, connecting the model and data components.",
      "prerequisites": [
        "GPT Configuration",
        "GPT Architecture",
        "Data Preparation & Loading"
      ],
      "key_source_files": [
        "train.py",
        "model.py"
      ],
      "key_symbols": [
        "configure_optimizers",
        "estimate_loss",
        "model.train",
        "model.eval"
      ],
      "learning_objectives": [
        "By the end, students will be able to implement a complete training loop for a language model.",
        "Students will understand the role of optimizers, learning rate schedulers, and evaluation metrics in model training.",
        "Students will learn how to save and load model checkpoints."
      ],
      "exercise_description": "Students will implement a `MiniTrainer` class that takes a model and data loader, then orchestrates the training, evaluation, and checkpointing process.",
      "visualization_idea": "A flowchart depicting the training loop: data batch -> forward pass -> loss -> backward pass -> optimizer step -> LR scheduler step -> evaluation -> repeat."
    },
    {
      "id": "05",
      "title": "Generating Text with a GPT Model",
      "concept": "Text Generation/Sampling",
      "description": "Students will learn how a trained GPT model generates new text, understanding the iterative prediction process and common sampling strategies like temperature and top-k filtering.",
      "prerequisites": [
        "GPT Architecture",
        "Training Orchestration"
      ],
      "key_source_files": [
        "sample.py",
        "model.py"
      ],
      "key_symbols": [
        "GPT.generate"
      ],
      "learning_objectives": [
        "By the end, students will be able to use a trained GPT model to generate coherent text.",
        "Students will understand the mechanisms of autoregressive text generation and token sampling techniques.",
        "Students will be able to implement `temperature` and `top_k` sampling."
      ],
      "exercise_description": "Students will extend their `MiniGPT` model with a `generate` method that takes a prompt and generates new tokens autoregressively, incorporating temperature-based sampling.",
      "visualization_idea": "An animated sequence showing the model predicting the next token, appending it to the input, and repeating, with a sidebar explaining sampling probabilities."
    }
  ],
  "capstone": {
    "title": "mini-nanoGPT",
    "description": "Students will integrate all learned concepts to build a simplified end-to-end version of nanoGPT. This includes defining a model configuration, constructing the GPT architecture, preparing a small dataset, training the model on that data, and finally generating new text with their trained mini-GPT. This project reinforces the entire lifecycle of a language model.",
    "estimated_hours": 8,
    "modules": [
      {
        "name": "MiniGPTConfig",
        "description": "A dataclass holding all essential hyperparameters for the MiniGPT model.",
        "depends_on": [],
        "interface_sketch": "from dataclasses import dataclass\n\n@dataclass\nclass MiniGPTConfig:\n    block_size: int = 256\n    vocab_size: int = 50257\n    n_layer: int = 6\n    n_head: int = 6\n    n_embd: int = 384\n    dropout: float = 0.1",
        "test_behaviors": [
          "Instantiating with defaults should create an object with correct default values.",
          "Instantiating with custom values should correctly assign them."
        ]
      },
      {
        "name": "MiniGPTModel",
        "description": "A simplified GPT model incorporating the transformer architecture, capable of forward passes and text generation.",
        "depends_on": [
          "MiniGPTConfig"
        ],
        "interface_sketch": "import torch.nn as nn\n\nclass MiniGPTModel(nn.Module):\n    def __init__(self, config: MiniGPTConfig):\n        super().__init__()\n        self.config = config\n        # ... layers defined here\n\n    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:\n        # Computes logits and optionally loss\n        ...\n\n    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:\n        # Generates new tokens autoregressively\n        ...",
        "test_behaviors": [
          "Calling forward with (batch_size, block_size) input should return (batch_size, block_size, vocab_size) logits.",
          "Calling generate with a prompt and max_new_tokens should produce an output tensor of correct length (prompt + max_new_tokens).",
          "get_num_parameters() method should return an integer count of trainable parameters."
        ]
      },
      {
        "name": "MiniTextDataLoader",
        "description": "Handles tokenization, data loading, and batching from a local text file.",
        "depends_on": [],
        "interface_sketch": "import torch\n\nclass MiniTextDataLoader:\n    def __init__(self, data_path: str, block_size: int, batch_size: int):\n        # ... tokenization and data preparation\n\n    def get_batch(self, split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:\n        # Returns a batch of (inputs, targets)\n        ...\n\n    def get_vocab_size(self) -> int:\n        # Returns the size of the vocabulary\n        ...",
        "test_behaviors": [
          "Calling `get_batch` should return two tensors of shape (batch_size, block_size).",
          "The returned tensors should contain valid token IDs within the vocabulary range.",
          "Calling `get_vocab_size` should return the correct number of unique tokens in the processed text."
        ]
      },
      {
        "name": "MiniTrainer",
        "description": "Manages the training and evaluation loop, including optimizer setup and learning rate scheduling.",
        "depends_on": [
          "MiniGPTModel",
          "MiniTextDataLoader"
        ],
        "interface_sketch": "import torch\n\nclass MiniTrainer:\n    def __init__(self, model: MiniGPTModel, data_loader: MiniTextDataLoader, config: dict):\n        # ... optimizer and scheduler setup\n\n    def train(self, num_iterations: int):\n        # Executes the full training loop\n        ...\n\n    def evaluate(self, num_batches: int = 10) -> float:\n        # Computes average loss on evaluation data\n        ...",
        "test_behaviors": [
          "Calling `train` should reduce the loss over iterations.",
          "Calling `evaluate` should return a float representing the average validation loss.",
          "The model's weights should change after calling `train`."
        ]
      }
    ],
    "integration_test": {
      "description": "Train a MiniGPTModel on a small subset of text data and verify it can generate coherent text that resembles the training data's style.",
      "setup_code": "import os\n\n# Create a dummy dataset file\ndummy_text = \"hello world hello world this is a test example. example test. world hello. new line.\\nhello world\"\nwith open(\"dummy_data.txt\", \"w\") as f:\n    f.write(dummy_text)\n\n# Define simplified configurations\nconfig = {\n    'block_size': 8,\n    'batch_size': 4,\n    'n_layer': 2,\n    'n_head': 2,\n    'n_embd': 32,\n    'max_iters': 20,\n    'learning_rate': 1e-3,\n    'eval_iters': 2\n}",
      "success_metric": "The generated text should contain tokens from the training vocabulary and exhibit some basic coherence, even if short. The training loss should decrease significantly over iterations.",
      "expected_output_check": "Print the training loss trajectory. After training, prompt the model with a starting phrase (e.g., 'hello') and print 20 generated tokens. Verify the generated output doesn't contain unknown tokens and includes words from the original `dummy_text`."
    }
  }
}
```