# Curriculum: Understanding nanoGPT

Source: https://github.com/karpathy/nanoGPT

## Overview

nanoGPT provides a clean, self-contained implementation for building and training GPT-style large language models. Its core is the `GPT` model, a standard Transformer decoder architecture comprising token embeddings, positional embeddings, a stack of `Block` modules (each with `CausalSelfAttention` and an `MLP`), and a final language modeling head. The `train.py` script manages the entire training lifecycle, from loading model configurations and efficiently prepared data to executing forward/backward passes and logging metrics.

## Notebooks

### 01. Defining GPT Model Hyperparameters with GPTConfig

Students will define a dataclass to manage all model hyperparameters, understanding how these values dictate the model's size and capabilities.

**Learning objectives:**
- By the end, students will be able to define and instantiate a `GPTConfig` object.
- Students will understand how hyperparameters like `n_layer`, `n_head`, and `n_embd` influence model complexity.

**Exercise:** Students will implement the `GPTConfig` dataclass from scratch, including default values for common hyperparameters.

### 02. Implementing Causal Self-Attention

Students will build the `CausalSelfAttention` mechanism, a core component for sequence modeling, learning how to mask future tokens to maintain causality.

**Learning objectives:**
- By the end, students will be able to implement a causal self-attention layer.
- Students will understand the role of causal masking in generative language models.

**Exercise:** Students will implement the `CausalSelfAttention` class, including query, key, value projections, attention score calculation, and causal masking.

### 03. Constructing the Transformer Block

Students will integrate the Causal Self-Attention with a Multi-Layer Perceptron (MLP) and apply Layer Normalization and residual connections to form a complete Transformer Block.

**Learning objectives:**
- By the end, students will be able to assemble a full Transformer Block.
- Students will understand the purpose of Layer Normalization and residual connections within a Transformer architecture.

**Exercise:** Students will implement the `Block` class, combining `CausalSelfAttention`, an MLP, Layer Normalization, and residual connections.

### 04. Assembling the Full GPT Model Architecture

Students will combine token embeddings, positional embeddings, and a stack of Transformer Blocks to construct the complete `GPT` model, including its language modeling head.

**Learning objectives:**
- By the end, students will be able to build the full GPT model from its constituent parts.
- Students will understand the role of token and positional embeddings in providing context to the model.

**Exercise:** Students will implement the `GPT` class, including embedding layers, the stack of `Block` modules, and the final language modeling head.

### 05. Preparing Text Data for GPT Training

Students will learn how to process raw text, tokenize it, and convert it into numerical sequences suitable for efficient loading and training of a GPT model.

**Learning objectives:**
- By the end, students will be able to tokenize raw text and convert it into numerical ID sequences.
- Students will understand the importance of efficient data loading strategies for large models.

**Exercise:** Students will write a simple Python script to load a text file, tokenize it using a character-level or simple BPE tokenizer, and save the resulting token IDs to a binary file.

### 06. Implementing the GPT Training Loop

Students will implement the core training loop responsible for initializing the model and optimizer, loading data batches, performing forward/backward passes, and updating model weights.

**Learning objectives:**
- By the end, students will be able to implement a complete training loop for a GPT model.
- Students will understand the interaction between the model, optimizer, and data loader during training.

**Exercise:** Students will write a basic training script that loads a pre-built GPT model, uses a dummy dataset, implements data batching, and performs forward and backward passes with a simple optimizer.

## Capstone Project

**mini-nanoGPT**

Students will build a simplified, end-to-end version of nanoGPT, integrating all previously learned components into a functional language model. This project emphasizes how data preparation, model architecture, and training loops come together to create a minimal yet complete system, highlighting the core principles behind modern LLMs.

---

<!-- Edit the JSON below to skip notebooks (set skip: true) -->

```json
{
  "title": "Understanding nanoGPT",
  "mental_model": "nanoGPT provides a clean, self-contained implementation for building and training GPT-style large language models. Its core is the `GPT` model, a standard Transformer decoder architecture comprising token embeddings, positional embeddings, a stack of `Block` modules (each with `CausalSelfAttention` and an `MLP`), and a final language modeling head. The `train.py` script manages the entire training lifecycle, from loading model configurations and efficiently prepared data to executing forward/backward passes and logging metrics.",
  "concepts": [
    {
      "name": "GPTConfig",
      "description": "A dataclass that defines all hyperparameters necessary to construct a GPT model, such as vocabulary size, block size, number of layers, heads, and embedding dimensions.",
      "complexity": "basic"
    },
    {
      "name": "CausalSelfAttention",
      "description": "The attention mechanism within a Transformer block that allows the model to weigh the importance of different tokens in a sequence, critically ensuring that a token can only attend to preceding tokens to maintain causality for language generation.",
      "complexity": "intermediate"
    },
    {
      "name": "Transformer Block",
      "description": "The fundamental repeating unit of the GPT model, consisting of a Causal Self-Attention layer followed by a Multi-Layer Perceptron (MLP), each preceded by Layer Normalization and residual connections.",
      "complexity": "basic"
    },
    {
      "name": "GPT Model Architecture",
      "description": "The overall structure of the GPT model, comprising token and positional embeddings, a stack of Transformer Blocks, and a final linear layer (language model head) for predicting the next token in the vocabulary. It also handles weight initialization and parameter tying.",
      "complexity": "intermediate"
    },
    {
      "name": "Data Preparation",
      "description": "The process of downloading, tokenizing, and converting raw text datasets (like Shakespeare or OpenWebText) into numerical token ID sequences, which are then stored in binary files for efficient loading during training.",
      "complexity": "basic"
    },
    {
      "name": "Training Loop",
      "description": "The main script responsible for orchestrating the entire training process: loading model configuration, initializing the GPT model, setting up the optimizer, loading data, performing forward and backward passes, and logging metrics.",
      "complexity": "intermediate"
    }
  ],
  "notebooks": [
    {
      "id": "01",
      "title": "Defining GPT Model Hyperparameters with GPTConfig",
      "concept": "GPTConfig",
      "description": "Students will define a dataclass to manage all model hyperparameters, understanding how these values dictate the model's size and capabilities.",
      "prerequisites": [],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "GPTConfig"
      ],
      "learning_objectives": [
        "By the end, students will be able to define and instantiate a `GPTConfig` object.",
        "Students will understand how hyperparameters like `n_layer`, `n_head`, and `n_embd` influence model complexity."
      ],
      "exercise_description": "Students will implement the `GPTConfig` dataclass from scratch, including default values for common hyperparameters.",
      "visualization_idea": "A textual representation of a `GPTConfig` instance showing its parameters and their values."
    },
    {
      "id": "02",
      "title": "Implementing Causal Self-Attention",
      "concept": "CausalSelfAttention",
      "description": "Students will build the `CausalSelfAttention` mechanism, a core component for sequence modeling, learning how to mask future tokens to maintain causality.",
      "prerequisites": [
        "01"
      ],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "CausalSelfAttention"
      ],
      "learning_objectives": [
        "By the end, students will be able to implement a causal self-attention layer.",
        "Students will understand the role of causal masking in generative language models."
      ],
      "exercise_description": "Students will implement the `CausalSelfAttention` class, including query, key, value projections, attention score calculation, and causal masking.",
      "visualization_idea": "Matplotlib heatmap visualizing the attention weights for a short input sequence, clearly showing the effect of causal masking."
    },
    {
      "id": "03",
      "title": "Constructing the Transformer Block",
      "concept": "Transformer Block",
      "description": "Students will integrate the Causal Self-Attention with a Multi-Layer Perceptron (MLP) and apply Layer Normalization and residual connections to form a complete Transformer Block.",
      "prerequisites": [
        "01",
        "02"
      ],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "Block"
      ],
      "learning_objectives": [
        "By the end, students will be able to assemble a full Transformer Block.",
        "Students will understand the purpose of Layer Normalization and residual connections within a Transformer architecture."
      ],
      "exercise_description": "Students will implement the `Block` class, combining `CausalSelfAttention`, an MLP, Layer Normalization, and residual connections.",
      "visualization_idea": "Graphviz block diagram illustrating the data flow within a single Transformer Block, including attention, MLP, normalization, and residual paths."
    },
    {
      "id": "04",
      "title": "Assembling the Full GPT Model Architecture",
      "concept": "GPT Model Architecture",
      "description": "Students will combine token embeddings, positional embeddings, and a stack of Transformer Blocks to construct the complete `GPT` model, including its language modeling head.",
      "prerequisites": [
        "01",
        "03"
      ],
      "key_source_files": [
        "model.py"
      ],
      "key_symbols": [
        "GPT.__init__",
        "GPT.forward"
      ],
      "learning_objectives": [
        "By the end, students will be able to build the full GPT model from its constituent parts.",
        "Students will understand the role of token and positional embeddings in providing context to the model."
      ],
      "exercise_description": "Students will implement the `GPT` class, including embedding layers, the stack of `Block` modules, and the final language modeling head.",
      "visualization_idea": "Graphviz diagram showing the overall `GPT` architecture, highlighting the embedding layers, the repeated `Block` modules, and the final output layer."
    },
    {
      "id": "05",
      "title": "Preparing Text Data for GPT Training",
      "concept": "Data Preparation",
      "description": "Students will learn how to process raw text, tokenize it, and convert it into numerical sequences suitable for efficient loading and training of a GPT model.",
      "prerequisites": [],
      "key_source_files": [
        "data/shakespeare_char/prepare.py",
        "data/openwebtext/prepare.py"
      ],
      "key_symbols": [
        "data.shakespeare_char.prepare.prepare",
        "data.openwebtext.prepare.prepare"
      ],
      "learning_objectives": [
        "By the end, students will be able to tokenize raw text and convert it into numerical ID sequences.",
        "Students will understand the importance of efficient data loading strategies for large models."
      ],
      "exercise_description": "Students will write a simple Python script to load a text file, tokenize it using a character-level or simple BPE tokenizer, and save the resulting token IDs to a binary file.",
      "visualization_idea": "Graphviz diagram illustrating the data pipeline: raw text -> tokenization -> numerical IDs -> splitting into train/validation sets -> saving to binary files."
    },
    {
      "id": "06",
      "title": "Implementing the GPT Training Loop",
      "concept": "Training Loop",
      "description": "Students will implement the core training loop responsible for initializing the model and optimizer, loading data batches, performing forward/backward passes, and updating model weights.",
      "prerequisites": [
        "04",
        "05"
      ],
      "key_source_files": [
        "train.py"
      ],
      "key_symbols": [
        "train.main",
        "train.get_batch"
      ],
      "learning_objectives": [
        "By the end, students will be able to implement a complete training loop for a GPT model.",
        "Students will understand the interaction between the model, optimizer, and data loader during training."
      ],
      "exercise_description": "Students will write a basic training script that loads a pre-built GPT model, uses a dummy dataset, implements data batching, and performs forward and backward passes with a simple optimizer.",
      "visualization_idea": "Matplotlib plot showing training and validation loss curves over epochs, demonstrating convergence and potential overfitting."
    }
  ],
  "capstone": {
    "title": "mini-nanoGPT",
    "description": "Students will build a simplified, end-to-end version of nanoGPT, integrating all previously learned components into a functional language model. This project emphasizes how data preparation, model architecture, and training loops come together to create a minimal yet complete system, highlighting the core principles behind modern LLMs.",
    "estimated_hours": 4,
    "modules": [
      {
        "name": "MiniGPT",
        "description": "A simplified GPT model, encapsulating the token embeddings, positional embeddings, a few Transformer Blocks, and the language modeling head.",
        "depends_on": [],
        "interface_sketch": "import torch.nn as nn\n\nclass MiniGPT(nn.Module):\n    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):\n        super().__init__()\n        # Embeddings, Blocks, LM head\n\n    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:\n        # Computes logits and optionally loss\n        ...",
        "test_behaviors": [
          "Given a dummy input tensor of shape (batch_size, block_size), the `forward` method should return logits of shape (batch_size, block_size, vocab_size).",
          "If `targets` are provided, the `forward` method should also return a scalar loss value.",
          "The causal attention mask should correctly prevent tokens from attending to future tokens."
        ]
      },
      {
        "name": "TextDataset",
        "description": "A class for loading and preparing text data, including tokenization and batching for training and validation splits.",
        "depends_on": [],
        "interface_sketch": "class TextDataset:\n    def __init__(self, raw_text: str, block_size: int, split_ratio: float = 0.9):\n        # Tokenize text, split into train/val\n\n    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:\n        # Returns (xb, yb) tensors for a given split\n        ...",
        "test_behaviors": [
          "Given a raw text string, the `get_batch('train', batch_size)` method should return two tensors of shape (batch_size, block_size).",
          "The `xb` and `yb` tensors should contain valid token IDs (integers within the vocabulary range).",
          "Successive calls to `get_batch` should return different, contiguous batches of data."
        ]
      },
      {
        "name": "MiniTrainer",
        "description": "Orchestrates the training process, including setting up the optimizer, managing data batches, and performing forward/backward passes for a specified number of training steps.",
        "depends_on": [
          "MiniGPT",
          "TextDataset"
        ],
        "interface_sketch": "class MiniTrainer:\n    def __init__(self, model: MiniGPT, dataset: TextDataset, learning_rate: float = 1e-3, device: str = 'cpu'):\n        # Initialize optimizer, model, dataset\n\n    def train(self, max_iters: int, eval_interval: int = 100):\n        # Main training loop with evaluation\n        ...",
        "test_behaviors": [
          "After calling `train` for a few iterations, the model's parameters should have updated (not be identical to initial state).",
          "The reported training loss should decrease over successive `eval_interval`s.",
          "The `train` method should complete without errors given valid inputs."
        ]
      }
    ],
    "integration_test": {
      "description": "Train a `MiniGPT` model on a small, synthesized text dataset for a limited number of iterations. Verify that the training loss decreases and that the model can generate a short, non-random sequence of characters, demonstrating basic learning.",
      "setup_code": "import torch\nimport random\n\n# Assuming MiniGPT, TextDataset, MiniTrainer are defined\n\n# 1. Synthesize a tiny text corpus\ncorpus = \"hello world this is a test and another test to make sure it learns. hello world again.\"\n\n# 2. Initialize dataset\nblock_size = 8\ndataset = TextDataset(corpus, block_size=block_size, split_ratio=0.7)\nvocab_size = len(set(list(corpus))) # Simplified for char-level\n\n# 3. Initialize model and trainer\nmodel = MiniGPT(vocab_size=vocab_size, block_size=block_size, n_layer=2, n_head=2, n_embd=32)\ntrainer = MiniTrainer(model, dataset, learning_rate=1e-3, device='cpu')\n\n# Capture initial state for comparison\ninitial_params = {n: p.clone() for n, p in model.named_parameters()}",
      "success_metric": "Training loss decreases by at least 10% after 100 iterations, and generated text shows some coherence (e.g., repeats patterns from the training corpus).",
      "expected_output_check": "The final training loss should be lower than the initial loss. Generate a short sequence (e.g., 20 tokens) by feeding the model a starting character's token ID and iteratively predicting the next token. Inspect the generated sequence for any recognizable words or patterns from the original corpus. The model's parameters should not be identical to their `initial_params`."
    }
  }
}
```