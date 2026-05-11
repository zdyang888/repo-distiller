# Capstone Project: mini-nanoGPT

By completing this project, you will have built a simplified, end-to-end version of nanoGPT from scratch. You will integrate data preparation, model architecture, and a training loop into a functional language model. This project demystifies the core principles behind modern LLMs by showing you how all the pieces fit together in a minimal, complete system.

- **Project:** mini-nanoGPT
- **Description:** Students will build a simplified, end-to-end version of nanoGPT, integrating all previously learned components into a functional language model. This project emphasizes how data preparation, model architecture, and training loops come together to create a minimal yet complete system, highlighting the core principles behind modern LLMs.
- **Estimated time:** 4 hours

---

## System Architecture

The project is composed of three main components: a dataset handler (`TextDataset`), the GPT model itself (`MiniGPT`), and a training orchestrator (`MiniTrainer`). The `MiniTrainer` coordinates the flow of data from the dataset to the model and updates the model's weights based on the calculated loss.

```ascii
+---------------+       get_batch()        +---------------+       forward()        +-----------+
|               |------------------------->|               |----------------------->|           |
|  TextDataset  | (xb, yb) training data   |  MiniTrainer  |   (logits, loss)     |  MiniGPT  |
|               |<-------------------------|               |<-----------------------|           |
+---------------+                          +---------------+    optimizer.step()    +-----------+
                                                 |                                     ^
                                                 | (updates model weights)             |
                                                 +-------------------------------------+
```

---

## Modules to Implement

You will implement three Python classes in `implementation.py`.

### 1. `TextDataset`

-   **Responsibility:** This class is responsible for loading a raw text corpus, creating a character-level vocabulary, tokenizing the text, and serving batches of data for training and validation. It handles the crucial first step of turning text into numbers the model can understand.

-   **Interface:**
    ```python
    class TextDataset:
        def __init__(self, raw_text: str, block_size: int, split_ratio: float = 0.9):
            # Tokenize text, split into train/val
            ...

        def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
            # Returns (xb, yb) tensors for a given split
            ...
    ```

-   **Expected Behavior:**
    -   Given a raw text string, the `get_batch('train', batch_size)` method should return two tensors, `xb` (inputs) and `yb` (targets), both of shape `(batch_size, block_size)`.
    -   The `xb` and `yb` tensors must contain valid token IDs (integers within the vocabulary range).
    -   Successive calls to `get_batch` should return different, contiguous batches of data, simulating the process of iterating through an epoch.

### 2. `MiniGPT`

-   **Responsibility:** This is the core of our language model. It encapsulates the token and positional embeddings, a stack of Transformer Blocks, and a final linear layer (the language modeling head) to produce predictions over the vocabulary. It must also correctly implement a causal attention mask to ensure it only uses past information to predict the next token.

-   **Interface:**
    ```python
    import torch
    import torch.nn as nn

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):
            super().__init__()
            # Implement token embeddings, positional embeddings, Transformer Blocks, and the LM head
            ...

        def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor | None]:
            # Computes logits and, optionally, the cross-entropy loss
            ...
    ```

-   **Expected Behavior:**
    -   Given a dummy input tensor `idx` of shape `(batch_size, block_size)`, the `forward` method should return logits of shape `(batch_size, block_size, vocab_size)`.
    -   If `targets` are provided during the forward pass, the method must also return a scalar loss value.
    -   The self-attention mechanism within the Transformer Blocks must use a causal mask to prevent any token from attending to future tokens.

### 3. `MiniTrainer`

-   **Responsibility:** This class orchestrates the entire training process. It holds the model, the dataset, and the optimizer. Its main `train` method runs the training loop: fetching batches, performing forward and backward passes, updating model weights, and periodically evaluating the model's performance.

-   **Interface:**
    ```python
    class MiniTrainer:
        def __init__(self, model: MiniGPT, dataset: TextDataset, learning_rate: float = 1e-3, device: str = 'cpu'):
            # Initialize optimizer, and store model and dataset references
            ...

        def train(self, max_iters: int, eval_interval: int = 100):
            # The main training loop: get batch, forward, backward, step, and evaluate
            ...
    ```

-   **Expected Behavior:**
    -   After `train` is called for a few iterations, the model's parameters should be updated (i.e., they are no longer identical to their initial state).
    -   The reported training loss should generally decrease over successive evaluation intervals, indicating that the model is learning.
    -   The `train` method must complete without errors when provided with a valid model and dataset.

---

## Connecting the Modules

The components are designed to work together sequentially:

1.  An instance of `TextDataset` is created from a source text.
2.  An instance of `MiniGPT` is created with the desired hyperparameters (including the `vocab_size` from the dataset).
3.  An instance of `MiniTrainer` is created, taking the `model` and `dataset` objects as arguments.
4.  Calling `trainer.train()` kicks off the process. The trainer repeatedly calls `dataset.get_batch()` to get input tensors (`xb`, `yb`) and passes them to `model.forward(xb, yb)` to get logits and loss. It then uses the loss to perform backpropagation and updates the model's weights.

---

## Success Criteria

Your implementation is successful when all the automated tests pass. The tests are designed to verify the "Expected Behavior" for each module and to run a final integration test.

The integration test will:
1.  Initialize your `TextDataset`, `MiniGPT`, and `MiniTrainer` with a small, predictable corpus.
2.  Run the training process for 100 iterations.
3.  **Verify that the training loss decreases by at least 10%.**
4.  **Verify that the model can generate a short, non-random sequence of characters, demonstrating that it has learned basic patterns from the data.**

---

## Suggested Implementation Order

We recommend implementing the components in the following order to manage dependencies effectively:

1.  **`TextDataset`**: Start with the data pipeline. It has no dependencies and is fundamental to everything else. You can test it in isolation to ensure your data loading and batching are correct.
2.  **`MiniGPT`**: Implement the model architecture next. This is the most complex piece. Focus on getting the dimensions right and correctly implementing the causal self-attention mask.
3.  **`MiniTrainer`**: Finally, implement the trainer. This component ties the dataset and model together and brings your project to life.

---

## Getting Started

All your code should be written in the `implementation.py` file. The tests in `test_capstone.py` will import your classes from that file to check them.

To begin working and run the tests, follow these steps:

1.  Navigate to the project directory.
    ```bash
    cd capstone/
    ```

2.  Open `implementation.py` in your editor and start building the classes described above.

3.  To run the tests and check your progress, execute `pytest` from the `capstone/` directory. Use the `-v` flag for verbose output.
    ```bash
    pytest test_capstone.py -v
    ```

The test suite will run, and you will see which tests pass and which fail. Use the test failures to guide your implementation until all tests pass. Good luck