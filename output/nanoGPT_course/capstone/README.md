# mini-nanoGPT: End-to-End LLM Pipeline

Welcome to your capstone project! This project is the culmination of everything you've learned, challenging you to build a complete, working pipeline for a small-scale Large Language Model (LLM).

**By completing this project, you will have built a complete, end-to-end pipeline for a Generative Pre-trained Transformer (GPT) model.** You will have implemented the entire process from raw text processing and tokenization to model definition, training, and text generation. This project synthesizes all the core concepts of deep learning and natural language processing covered in the course into a single, tangible application.

**Estimated Time:** 4 hours

## System Architecture

The project is divided into three main components that work together to process data, train a model, and generate text. The diagram below illustrates the flow of data through the system for both training and inference.

```ascii
                                     TRAINING
            +---------------------+
Raw Text -> | MiniDataProcessor | -> train.bin / val.bin
            +---------------------+
                    |
                    | (File path)
                    v
            +---------------------+       +----------------+
            |    MiniTrainer    | ----> | MiniGPTModel   | (forward pass for loss)
            +---------------------+       +----------------+
                    |                             ^
                    | (Saves model)               | (instantiates)
                    v                             |
            checkpoint.pt ------------------------+

-----------------------------------------------------------------------------------

                                     INFERENCE
            +---------------------+
            |   checkpoint.pt   | (Loads trained model)
            +---------------------+
                    |
                    v
            +----------------+
Prompt ->   | MiniGPTModel   | -> Generated Text
            +----------------+
              (generate call)
```

## Modules to Implement

You will implement three Python classes in `implementation.py`. The tests in `test_capstone.py` will verify that each class meets the required specifications.

### 1. `MiniDataProcessor`

This module is the starting point of our pipeline. It takes raw text and prepares it for the model.

*   **Responsibility:** Handles character-level tokenization of raw text and serializes the tokenized data into a binary file. This format is efficient for the trainer to read during training.
*   **Interface Sketch:**
    ```python
    class MiniDataProcessor:
        def __init__(self, vocab_size: int):
            # Should initialize token encoders/decoders
            ...
        def prepare_data(self, text: str, output_file: str) -> None:
            # Should tokenize text and write it to the binary output_file
            ...
    ```
*   **Expected Behavior (what the tests check):**
    *   Given a short string of text, `prepare_data` successfully creates a `.bin` file at the specified path.
    *   The contents of the `.bin` file must be the correct numerical (integer) representations of the tokens from the input text.

### 2. `MiniGPTModel`

This is the core of our project—the neural network itself. It's a simplified version of a GPT model, but it contains all the essential components of the transformer architecture.

*   **Responsibility:** Defines the GPT architecture, including token and position embeddings, transformer blocks, and the final prediction head. It must implement both a `forward` pass for training and a `generate` method for inference.
*   **Interface Sketch:**
    ```python
    import torch
    import torch.nn as nn
    from typing import Optional, Tuple

    class MiniGPTModel(nn.Module):
        def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):
            super().__init__()
            ...
        def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            # Process input `idx` and return logits. If `targets` are provided, also compute and return the loss.
            ...
        def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
            # Take a conditioning sequence `idx` and generate `max_new_tokens` more.
            ...
    ```
*   **Expected Behavior:**
    *   Calling `forward` with an input tensor `idx` should return a tensor of logits with the correct shape. If `targets` are also provided, it should return a tuple of (logits, loss).
    *   Calling `generate` with a starting sequence `idx` should produce a new tensor that extends the original sequence by `max_new_tokens`.

### 3. `MiniTrainer`

This module orchestrates the training process. It connects the data and the model to actually learn from the data.

*   **Responsibility:** Manages the training loop. This includes loading data in batches, feeding batches to the model, calculating loss, performing backpropagation with an optimizer, and saving the trained model weights (checkpointing).
*   **Interface Sketch:**
    ```python
    class MiniTrainer:
        def __init__(self, model: MiniGPTModel, train_data_file: str, val_data_file: str, batch_size: int, learning_rate: float, max_iters: int):
            # Initialize optimizer, data loaders, etc.
            ...
        def train(self) -> None:
            # Run the main training loop for `max_iters`.
            ...
    ```
*   **Expected Behavior:**
    *   Running the `train` method should cause the model's validation loss to decrease over a small number of iterations, indicating that learning is occurring.
    *   After the `train` method completes, a model checkpoint file (e.g., `latest_checkpoint.pt`) should be saved to disk.

## How It All Fits Together: The Integration Test

The final test (`test_end_to_end`) ensures all your modules work together correctly. It simulates the entire pipeline:

1.  **Setup:** A small, predictable text corpus is defined: `"the quick brown fox jumps over the lazy dog. the lazy dog sleeps."`
2.  **Data Processing:** Your `MiniDataProcessor` is used to tokenize this text and save it to `tiny_data.bin`.
3.  **Model & Trainer Init:** Your `MiniGPTModel` and `MiniTrainer` are instantiated with a small configuration.
4.  **Training:** The `MiniTrainer.train()` method is called to train the model on this tiny dataset for 100 iterations.
5.  **Inference:** The saved checkpoint is loaded back into the model. The model is then prompted with the tokens for `"the quick"`.
6.  **Validation:** The test checks if the text generated by the model contains plausible words from the training data (e.g., "brown", "fox", "dog"). This proves that the model has actually learned patterns from the data.

## Success Criteria

Your project is considered complete and successful when all automated tests pass. You can run the tests with `pytest`. A passing test suite demonstrates that each module fulfills its contract and that they integrate correctly to form a functional LLM pipeline.

## Suggested Implementation Order

We strongly recommend implementing the modules in the order of their dependencies to allow for incremental testing and development.

1.  **`MiniDataProcessor`:** Start here. It's self-contained and is the first step in the pipeline. Make sure you can correctly tokenize text and write it to a binary file.
2.  **`MiniGPTModel`:** This is the most complex piece. Focus on building the transformer architecture. Don't worry about training yet; just ensure the `forward` and `generate` methods produce tensors of the expected shape.
3.  **`MiniTrainer`:** Finally, implement the trainer. This will bring your data processor and model together. Your main goal here is to create a training loop that successfully reduces the model's loss.

## Getting Started

All your work should be done in the `implementation.py` file.

To begin, navigate to the project directory and run the test suite. You will see all tests failing. Your goal is to implement the classes until all tests pass.

```bash
# Navigate to the capstone project directory
cd capstone/

# Implement the MiniDataProcessor, MiniGPTModel, and MiniTrainer classes
# in the provided `implementation.py` file.

# Run the tests to check your work. Use -v for verbose output.
pytest test_capstone.py -v
```

Good luck