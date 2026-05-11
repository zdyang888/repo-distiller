"""
Module: interfaces.py
Author: A.I. Assistant
Date: 2023-10-27

Description:
This module defines the abstract base classes (ABCs) that form the core
architectural contract for the MiniGPT capstone project. By defining these
interfaces, we establish a clear separation of concerns and ensure that different
components of the system (data loading, modeling, training) can be developed
and tested independently, as long as they adhere to these contracts.

This approach facilitates modularity, testability, and potential future
extensions, such as swapping out different model architectures or data loaders.

The defined interfaces are:
- MiniGPTConfig: A dataclass for storing model hyperparameters.
- IMiniGPTModel: An ABC for the core GPT model, defining forward pass,
  text generation, and parameter counting.
- IMiniTextDataLoader: An ABC for the data handling component, responsible
  for tokenizing text, creating batches, and providing vocabulary info.
- IMiniTrainer: An ABC for the training orchestration, managing the
  training loop, optimization, and evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Any

import torch
import torch.nn as nn


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class MiniGPTConfig:
    """
    A dataclass holding all essential hyperparameters for the MiniGPT model.
    This object is passed to the model during instantiation to configure its
    architecture.
    """
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1


# -----------------------------------------------------------------------------
# Core Component Interfaces (ABCs)
# -----------------------------------------------------------------------------

class IMiniGPTModel(nn.Module, ABC):
    """
    Abstract Base Class for a simplified GPT model.

    This interface defines the essential contract for any model implementation.
    It must be a torch.nn.Module and support a forward pass for training,
    an autoregressive generation method for inference, and a utility for
    counting its parameters.
    """

    @abstractmethod
    def __init__(self, config: MiniGPTConfig):
        """
        Initializes the model architecture based on the provided configuration.

        CONTRACT:
        - Must call super().__init__().
        - Must store the configuration object.
        - Must construct all necessary layers (e.g., token embeddings,
          positional embeddings, transformer blocks, final layer norm, and
          the language model head).
        """
        super().__init__()
        # TODO: Implement this
        raise NotImplementedError("The __init__ method must be implemented by a subclass.")

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through the model.

        CONTRACT:
        - `idx` is a tensor of shape (batch_size, block_size) containing token indices.
        - `targets` (optional) is a tensor of the same shape as `idx`, used for
          calculating the loss.
        - Must return a tuple containing:
          1. `logits`: A tensor of shape (batch_size, block_size, vocab_size).
          2. `loss`: A scalar tensor (torch.Tensor) representing the cross-entropy
             loss if `targets` are provided, otherwise None.
        """
        # TODO: Implement this
        raise NotImplementedError("The forward pass must be implemented by a subclass.")

    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generates new tokens autoregressively.

        CONTRACT:
        - `idx` is a tensor of shape (batch_size, current_sequence_length)
          representing the initial context or prompt.
        - `max_new_tokens` is the number of new tokens to generate.
        - Must loop `max_new_tokens` times, each time:
          1. Conditioning on the current sequence `idx`.
          2. Getting model predictions (logits).
          3. Sampling a new token from the distribution (applying temperature
             and top-k sampling).
          4. Appending the new token to the sequence.
        - Must return a tensor of shape (batch_size, current_sequence_length + max_new_tokens).
        """
        # TODO: Implement this
        raise NotImplementedError("The generate method must be implemented by a subclass.")

    @abstractmethod
    def get_num_parameters(self) -> int:
        """
        Calculates and returns the total number of trainable parameters in the model.

        CONTRACT:
        - Must iterate through all model parameters (self.parameters()).
        - Must only count parameters where `requires_grad` is True.
        - Must return a single integer representing the total count.
        """
        # TODO: Implement this
        raise NotImplementedError("The get_num_parameters method must be implemented by a subclass.")


class IMiniTextDataLoader(ABC):
    """
    Abstract Base Class for a data loader.

    This interface defines the contract for handling data preparation, which
    includes reading a text file, tokenizing it, and serving batches of data
    for training and validation.
    """

    @abstractmethod
    def __init__(self, data_path: str, block_size: int, batch_size: int):
        """
        Initializes the data loader.

        CONTRACT:
        - Must read the text data from `data_path`.
        - Must create a vocabulary (character-level or using a pre-trained tokenizer).
        - Must tokenize the entire dataset into a single sequence of integers.
        - Must split the data into training and validation sets (e.g., 90%/10%).
        """
        # TODO: Implement this
        raise NotImplementedError("The __init__ method must be implemented by a subclass.")

    @abstractmethod
    def get_batch(self, split: str = 'train') -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single batch of inputs and targets.

        CONTRACT:
        - `split` must be either 'train' or 'val'.
        - Must randomly sample starting positions from the appropriate data split.
        - Must create an input tensor `x` of shape (batch_size, block_size).
        - Must create a target tensor `y` of shape (batch_size, block_size), where
          `y` is the sequence in `x` shifted by one position.
        - Must return the tuple (x, y).
        """
        # TODO: Implement this
        raise NotImplementedError("The get_batch method must be implemented by a subclass.")

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Returns the size of the vocabulary.

        CONTRACT:
        - Must return the total number of unique tokens in the vocabulary.
        """
        # TODO: Implement this
        raise NotImplementedError("The get_vocab_size method must be implemented by a subclass.")


class IMiniTrainer(ABC):
    """
    Abstract Base Class for the model trainer.

    This interface defines the contract for orchestrating the training and
    evaluation loops. It is responsible for managing the optimizer, running
    training iterations, and computing evaluation metrics.
    """

    @abstractmethod
    def __init__(self, model: IMiniGPTModel, data_loader: IMiniTextDataLoader, config: dict[str, Any]):
        """
        Initializes the trainer.

        CONTRACT:
        - Must accept a model, a data loader, and a configuration dictionary.
        - The `config` dictionary should contain training-specific hyperparameters
          like learning rate, weight decay, number of epochs/iterations, etc.
        - Must set up the optimizer (e.g., AdamW) and potentially a learning
          rate scheduler.
        - Must move the model to the appropriate device (e.g., 'cuda' or 'cpu').
        """
        # TODO: Implement this
        raise NotImplementedError("The __init__ method must be implemented by a subclass.")

    @abstractmethod
    def train(self, num_iterations: int) -> None:
        """
        Executes the main training loop.

        CONTRACT:
        - Must loop for `num_iterations`.
        - In each iteration, it must:
          1. Fetch a batch of data using `data_loader.get_batch('train')`.
          2. Perform a forward and backward pass.
          3. Update the model's weights using the optimizer.
          4. Zero out the gradients.
        - May include logging of training loss and other metrics.
        - The model's weights must be updated after this method is called.
        """
        # TODO: Implement this
        raise NotImplementedError("The train method must be implemented by a subclass.")

    @abstractmethod
    def evaluate(self, num_batches: int = 10) -> float:
        """
        Computes the average loss on the validation set.

        CONTRACT:
        - Must set the model to evaluation mode (e.g., `model.eval()`).
        - Must run the evaluation loop for `num_batches`.
        - Must compute the loss for each batch without performing backpropagation
          (e.g., within a `torch.no_grad()` context).
        - Must return a single float representing the average validation loss
          across all evaluated batches.
        - Must set the model back to training mode (e.g., `model.train()`) before exiting.
        """
        # TODO: Implement this
        raise NotImplementedError("The evaluate method must be implemented by a subclass.")