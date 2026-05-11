Here is a detailed capstone project README based on your specifications.

---

# Capstone Project: mini-nanoGPT

Welcome to your final capstone project! This project is designed to be a comprehensive exercise, integrating all the major concepts you've learned about building and training neural networks.

**By completing this project, you will have built a simplified, end-to-end version of a GPT-style language model.** You will have implemented the core components yourself: the model configuration, the transformer architecture, the data loading pipeline, and the training loop. You'll finish by generating novel text from a model you trained from scratch, giving you a solid, practical understanding of the entire lifecycle of a modern language model.

**Estimated Time to Complete:** 8 hours

## 1. Project Overview

The goal of this project is to build `mini-nanoGPT`, a small-scale implementation of a Generative Pre-trained Transformer. You will implement four key modules that work together to load text data, define a model, train it, and generate new text. This project will test your understanding of PyTorch, model architecture, data handling, and training dynamics.

### System Architecture

The project is composed of four main components. The `MiniTrainer` orchestrates the entire process, using the `MiniTextDataLoader` to get data and the `MiniGPTModel` (configured by `MiniGPTConfig`) to make predictions and learn.

```ascii
                  +-----------------+
                  | MiniGPTConfig   |
                  | (Hyperparams)   |
                  +-------+---------+
                          |
                          | (configures)
                          v
+--------------------+  +-----------------+  +------------------+
| MiniTextDataLoader |  |  MiniGPTModel   |  |   Optimizer &    |
| (Loads/batches     |  | (Architecture)  |  |   Scheduler    |
|  text data)        |  +-------+---------+  +--------+---------+
+---------+----------+          ^                    ^
          |                     | (updates weights)  | (used by)
          | (provides data)     |                    |
          v                     +--------------------+
+-------------------------------------------------------------+
|                         MiniTrainer                         |
|      (Orchestrates training loop, evaluation, logging)      |
+-------------------------------------------------------------+
```

## 2. Modules to Implement

You will implement the following four Python classes in the `implementation.py` file. Each class has a specific role, and the provided tests will verify its behavior.

### A. `MiniGPTConfig`

*   **Responsibility**: This is a simple data class that holds all the essential hyperparameters for the model. It acts as a single, organized source of truth for the model's architecture and training configuration.
*   **Interface Sketch**:
    ```python
    from dataclasses import dataclass

    @dataclass
    class MiniGPTConfig:
        block_size: int = 256
        vocab_size: int = 50257
        n_layer: int = 6
        n_head: int = 6
        n_embd: int = 384
        dropout: float = 0.1
    ```
*   **Test Behaviors**:
    *   Instantiating with defaults should create an object with correct default values.
    *   Instantiating with custom values should correctly assign them.

### B. `MiniTextDataLoader`

*   **Responsibility**: This class handles all data-related tasks. It reads a raw text file, creates a vocabulary, tokenizes the text into integers, and provides batches of data for training and validation.
*   **Interface Sketch**:
    ```python
    import torch

    class MiniTextDataLoader:
        def __init__(self, data_path: str, block_size: int, batch_size: int):
            # ... tokenization and data preparation

        def get_batch(self, split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
            # Returns a batch of (inputs, targets)

        def get_vocab_size(self) -> int:
            # Returns the size of the vocabulary
    ```
*   **Test Behaviors**:
    *   Calling `get_batch` should return two tensors of shape `(batch_size, block_size)`.
    *   The returned tensors should contain valid token IDs within the vocabulary range.
    *   Calling `get_vocab_size` should return the correct number of unique characters in the input text.

### C. `MiniGPTModel`

*   **Responsibility**: This is the core of our project. This class defines the GPT architecture using PyTorch modules, including the token and positional embeddings, transformer blocks, and the final prediction head. It must be able to perform a forward pass to calculate logits (and loss) and generate new text autoregressively.
*   **Interface Sketch**:
    ```python
    import torch
    import torch.nn as nn

    class MiniGPTModel(nn.Module):
        def __init__(self, config: MiniGPTConfig):
            super().__init__()
            self.config = config
            # ... layers defined here

        def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
            # Computes logits and optionally loss
            ...

        def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
            # Generates new tokens autoregressively
            ...
    ```
*   **Test Behaviors**:
    *   Calling `forward` with an input tensor of shape `(batch_size, block_size)` should return logits of shape `(batch_size, block_size, vocab_size)`.
    *   Calling `generate` with a prompt and `max_new_tokens` should produce an output tensor of shape `(batch_size, prompt_len + max_new_tokens)`.
    *   A helper method `get_num_parameters()` should return an integer count of trainable parameters.

### D. `MiniTrainer`

*   **Responsibility**: This class manages the entire training and evaluation process. It initializes the optimizer (e.g., AdamW), runs the training loop for a specified number of iterations, evaluates the model's performance on a validation set, and logs the results.
*   **Interface Sketch**:
    ```python
    import torch

    class MiniTrainer:
        def __init__(self, model: MiniGPTModel, data_loader: MiniTextDataLoader, config: dict):
            # ... optimizer and scheduler setup

        def train(self, num_iterations: int):
            # Executes the full training loop

        def evaluate(self, num_batches: int = 10) -> float:
            # Computes average loss on evaluation data
    ```
*   **Test Behaviors**:
    *   Calling `train` should reduce the model's loss over a number of iterations.
    *   Calling `evaluate` should return a float representing the average validation loss.
    *   The model's weights should have different values after `train` is called compared to before.

## 3. How the Modules Connect

*   `MiniGPTConfig` is instantiated and passed to the `__init__` method of **`MiniGPTModel`** to define its architecture.
*   `MiniTextDataLoader` is instantiated and passed to the `__init__` method of **`MiniTrainer`** to provide data.
*   `MiniGPTModel` is instantiated and passed to the `__init__` method of **`MiniTrainer`**, which will be responsible for updating its weights.
*   Inside the `MiniTrainer.train()` loop, the trainer will call `data_loader.get_batch()` to get inputs and targets, then pass them to `model.forward()` to get the loss, and finally use the optimizer to perform a backpropagation step.

## 4. Suggested Implementation Order

To make development easier, we recommend implementing the modules in an order that respects their dependencies.

1.  **`MiniGPTConfig`**: Start here. It's the simplest and has no dependencies.
2.  **`MiniTextDataLoader`**: This can be implemented next, as it also has no project dependencies. Getting the data pipeline right early is crucial.
3.  **`MiniGPTModel`**: Implement the model architecture. This is the most complex part. You'll need a working `MiniGPTConfig` to initialize it. Focus on getting the `forward` pass correct first.
4.  **`MiniTrainer`**: Finally, implement the trainer. It depends on all other components. This will tie everything together into a functional training loop.

## 5. Success Criteria & Integration Test

Your primary goal is to **make all tests pass**. The `pytest` suite is designed to test each module in isolation and then together.

The final test is an **integration test** that verifies the end-to-end functionality of your system.

*   **Description**: This test trains your `MiniGPTModel` on a small, dummy text file and verifies that it can learn and generate coherent text.
*   **Setup**: A small text file (`dummy_data.txt`) is created with repetitive text like `"hello world hello world..."`. A lightweight model and trainer configuration is used to ensure the test runs quickly.
*   **Success Metric**:
    1.  The training loss reported by the `MiniTrainer` must decrease significantly over the training iterations.
    2.  The text generated by the trained model should be coherent (for its small world) and only contain tokens present in the training vocabulary.
*   **Output Check**: The test will print the training loss at various steps. After training, it will prompt the model with a starting word (e.g., `'hello'`) and print the generated text. The output should consist of words from the original dummy text.

## 6. Getting Started

1.  Navigate to the project directory.
    ```bash
    cd capstone/
    ```

2.  Open `implementation.py` in your editor. This is the only file you need to modify. All the classes and methods described above are stubbed out for you.

3.  Implement the classes one by one, following the suggested order.

4.  Run the tests from your terminal to check your progress. The `-v` flag provides verbose output, which is helpful for debugging.
    ```bash
    pytest test_capstone.py -v
    ```

Good luck