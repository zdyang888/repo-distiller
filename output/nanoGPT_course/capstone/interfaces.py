"""
Defines the abstract interfaces for a capstone project building a miniature GPT.

This file establishes the core contracts for the three main components of the
system: the model, the dataset, and the trainer. By using Abstract Base Classes
(ABCs), we enforce a clear separation of concerns and ensure that any concrete
implementation will adhere to a consistent API.

The architecture is as follows:
- ITextDataset: An interface for a class responsible for loading, tokenizing,
  and batching text data. It provides the vocabulary and serves data for
  training and validation.
- IMiniGPT: An interface for the language model itself, a simplified version of
  GPT. It must be a torch.nn.Module that can perform a forward pass to compute
  logits and loss.
- IMiniTrainer: An interface for the training orchestrator. It takes a model
  and a dataset, and manages the training loop, including optimization, loss
- calculation, and periodic evaluation.
"""
from abc import ABC, abstractmethod
from collections.abc import Callable

import torch
import torch.nn as nn


class IMiniGPT(nn.Module, ABC):
    """
    Abstract Base Class for a simplified GPT model.

    This interface defines the essential structure of the language model.
    Any class that implements this interface must be a `torch.nn.Module`
    and provide a `forward` method that can compute logits and, optionally,
    the cross-entropy loss.
    """

    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):
        """
        Initializes the MiniGPT model architecture.

        This constructor's signature is part of the contract, but its
        implementation is left to the concrete class.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
            block_size (int): The maximum sequence length (context size).
            n_layer (int): The number of Transformer blocks.
            n_head (int): The number of attention heads in each Transformer block.
            n_embd (int): The dimensionality of the token and position embeddings.
        """
        super().__init__()
        # TODO: Implement this
        # The concrete implementation should initialize the following layers:
        # - Token embedding layer
        # - Positional embedding layer
        # - A sequence of Transformer blocks
        # - A final layer normalization
        # - The language modeling head (a linear layer to map to vocab size)
        # Note: While __init__ cannot be abstract, subclasses must adhere to this signature.

    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Performs the forward pass of the model.

        CONTRACT:
        Given an input tensor `idx` of token indices, this method must compute the
        logits for the next token prediction. If `targets` are provided, it must
        also compute and return the cross-entropy loss. The causal attention
        mask must be correctly applied to prevent tokens from attending to
        future positions.

        Args:
            idx (torch.Tensor): A tensor of token indices.
                Shape: (batch_size, block_size)
            targets (torch.Tensor | None, optional): A tensor of target token indices
                for loss calculation. If None, loss is not computed.
                Shape: (batch_size, block_size)

        Returns:
            tuple[torch.Tensor, torch.Tensor | None]:
            - logits (torch.Tensor): The model's output logits.
              Shape: (batch_size, block_size, vocab_size)
            - loss (torch.Tensor | None): A scalar tensor representing the
              cross-entropy loss if `targets` were provided, otherwise None.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the forward pass.")


class ITextDataset(ABC):
    """
    Abstract Base Class for a text data-handling class.

    This interface defines the contract for loading raw text, tokenizing it,
    splitting it into training and validation sets, and providing batches
    of data for model training.
    """

    def __init__(self, raw_text: str, block_size: int, split_ratio: float = 0.9):
        """
        Initializes the dataset.

        This constructor's signature is part of the contract. The implementation
        should process the raw text to build a vocabulary and split the tokenized
        data into training and validation sets.

        Args:
            raw_text (str): The full corpus of text to process.
            block_size (int): The sequence length for each batch.
            split_ratio (float): The proportion of data to use for the training set.
        """
        # TODO: Implement this
        # The concrete implementation should:
        # 1. Create a character-level or subword-level vocabulary.
        # 2. Implement an encoder (string to list[int]) and a decoder (list[int] to string).
        # 3. Tokenize the entire raw_text.
        # 4. Split the tokenized data into train and validation sets based on split_ratio.
        # Note: While __init__ cannot be abstract, subclasses must adhere to this signature.

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the total number of unique tokens in the vocabulary."""
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the vocab_size property.")

    @property
    @abstractmethod
    def encoder(self) -> Callable[[str], list[int]]:
        """
        Returns the encoder function.

        The encoder function should take a string and return a list of
        corresponding token IDs.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the encoder property.")

    @property
    @abstractmethod
    def decoder(self) -> Callable[[list[int]], str]:
        """
        Returns the decoder function.

        The decoder function should take a list of token IDs and return the
        corresponding string.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the decoder property.")

    @abstractmethod
    def get_batch(self, split: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a random batch of data from the specified split.

        CONTRACT:
        This method must sample `batch_size` number of random starting points
        from the specified data split ('train' or 'val'). For each starting
        point, it must extract a sequence of `block_size` tokens for the input
        (`xb`) and a corresponding sequence of `block_size` tokens for the target
        (`yb`), where the target is shifted by one position.

        Args:
            split (str): The data split to sample from. Must be 'train' or 'val'.
            batch_size (int): The number of independent sequences in the batch.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
            - xb (torch.Tensor): The input sequences.
              Shape: (batch_size, block_size)
            - yb (torch.Tensor): The target sequences (shifted by one).
              Shape: (batch_size, block_size)
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement batch generation.")


class IMiniTrainer(ABC):
    """
    Abstract Base Class for the training orchestrator.

    This interface defines the contract for a class that manages the model
    training process. It is responsible for setting up the optimizer, running
    the training loop, and performing periodic evaluations.
    """

    def __init__(self, model: IMiniGPT, dataset: ITextDataset, learning_rate: float = 1e-3, device: str = 'cpu'):
        """
        Initializes the trainer.

        This constructor's signature is part of the contract. The implementation
        should store the model and dataset, move the model to the specified
        device, and initialize an optimizer (e.g., AdamW).

        Args:
            model (IMiniGPT): The model instance to be trained.
            dataset (ITextDataset): The dataset instance to provide data.
            learning_rate (float): The learning rate for the optimizer.
            device (str): The device to train on ('cpu', 'cuda', etc.).
        """
        # TODO: Implement this
        # The concrete implementation should:
        # 1. Store references to the model and dataset.
        # 2. Move the model to the specified device.
        # 3. Instantiate an optimizer (e.g., torch.optim.AdamW).
        # Note: While __init__ cannot be abstract, subclasses must adhere to this signature.

    @abstractmethod
    def train(self, max_iters: int, eval_interval: int = 100, batch_size: int = 32) -> None:
        """
        Runs the main training loop.

        CONTRACT:
        This method must orchestrate the training process for a specified number
        of iterations (`max_iters`). In each iteration, it must:
        1. Fetch a batch of data using the dataset's `get_batch` method.
        2. Perform a forward and backward pass to compute gradients.
        3. Update the model's weights using the optimizer.
        Periodically (every `eval_interval` iterations), it should evaluate the
        model's performance on a validation set and report the loss. The model's
        parameters should be demonstrably updated after training.

        Args:
            max_iters (int): The total number of training iterations to run.
            eval_interval (int): The frequency (in iterations) at which to
                perform and report evaluation.
            batch_size (int): The batch size to use for training and evaluation.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the training loop.")