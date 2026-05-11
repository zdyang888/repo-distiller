"""
Interfaces for the Mini-GPT Capstone Project.

This file defines the abstract base classes (ABCs) for the core components
of the mini-GPT project. These interfaces establish a clear contract for each
module's functionality, ensuring that different implementations can be used
interchangeably as long as they adhere to these contracts.

The architecture is composed of three main parts:
1.  IMiniDataProcessor: Responsible for taking raw text data and converting it
    into a numerical format that the model can understand.
2.  IMiniGPTModel: The core transformer-based language model, responsible for
    learning patterns in the data and generating new text.
3.  IMiniTrainer: Manages the entire training process, orchestrating the data
    and the model to train the model's parameters effectively.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn


class IMiniDataProcessor(ABC):
    """
    Handles tokenization of raw text and its conversion to a binary format.

    This interface defines the contract for any data processing component.
    Its primary responsibility is to create a vocabulary from a text corpus
    and then encode that corpus into a sequence of integer token IDs, which are
    saved to a binary file for efficient loading during training.
    """

    @abstractmethod
    def __init__(self, vocab_size: int):
        """
        Initializes the data processor.

        Args:
            vocab_size (int): The maximum number of unique tokens in the
                              vocabulary.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the __init__ method.")

    @abstractmethod
    def prepare_data(self, text: str, output_file: str) -> None:
        """
        Processes raw text and saves the tokenized data to a binary file.

        CONTRACT:
        - Must build a vocabulary from the input `text`.
        - Must tokenize and encode the entire `text` into a sequence of
          integer token IDs based on the built vocabulary.
        - Must serialize and write the encoded token IDs to the specified
          `output_file` in a binary format (e.g., as a sequence of uint16).

        Args:
            text (str): The raw input text corpus.
            output_file (str): The path to the output `.bin` file where the
                               encoded data will be saved.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the prepare_data method.")


class IMiniGPTModel(nn.Module, ABC):
    """
    A simplified GPT model with a core transformer architecture.

    This interface defines the contract for the language model itself. It must
    be a valid `torch.nn.Module` and provide methods for both training (forward
    pass with loss calculation) and inference (text generation).
    """

    @abstractmethod
    def __init__(self, vocab_size: int, block_size: int, n_layer: int, n_head: int, n_embd: int):
        """
        Initializes the model architecture and its parameters.

        Args:
            vocab_size (int): The size of the vocabulary.
            block_size (int): The context length (max sequence length).
            n_layer (int): The number of transformer blocks.
            n_head (int): The number of attention heads.
            n_embd (int): The embedding dimension.
        """
        super().__init__()
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the __init__ method.")

    @abstractmethod
    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performs a forward pass through the model.

        CONTRACT:
        - Must take a batch of input token sequences `idx` of shape
          (batch_size, block_size).
        - Must return logits, which are the raw, unnormalized predictions for
          the next token in the sequence, with shape
          (batch_size, block_size, vocab_size).
        - If `targets` (of shape (batch_size, block_size)) are provided, it
          must also compute the cross-entropy loss between the logits and
          the targets.
        - The return value must be a tuple: (logits, loss). If `targets` is
          None, the loss component of the tuple should also be None.

        Args:
            idx (torch.Tensor): A tensor of input token indices.
            targets (Optional[torch.Tensor]): A tensor of target token indices
                                              for loss calculation.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing
            the model's output logits and the calculated loss (if any).
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """
        Generates a sequence of new tokens given a starting context.

        CONTRACT:
        - Must take a starting context `idx` of shape (batch_size, sequence_length).
        - Must autoregressively generate `max_new_tokens` by repeatedly
          calling the forward pass, sampling from the output logits, and
          appending the result to the input sequence.
        - Must ensure the context window does not exceed `block_size` by
          cropping the sequence if necessary.
        - Must return the extended sequence of tokens, including the original
          context `idx`, with a final shape of
          (batch_size, sequence_length + max_new_tokens).

        Args:
            idx (torch.Tensor): The initial sequence of token indices (context).
            max_new_tokens (int): The number of new tokens to generate.

        Returns:
            torch.Tensor: The generated sequence of token indices, including
                          the original context.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the generate method.")


class IMiniTrainer(ABC):
    """
    Manages the training loop for the MiniGPTModel.

    This interface defines the contract for the training component. It is
    responsible for all aspects of the training process, including creating
    data batches, running the optimization steps, evaluating the model on a
    validation set, and saving model checkpoints.
    """

    @abstractmethod
    def __init__(self, model: IMiniGPTModel, train_data_file: str, val_data_file: str, batch_size: int, learning_rate: float, max_iters: int):
        """
        Initializes the trainer with the model and training configuration.

        Args:
            model (IMiniGPTModel): The model instance to be trained.
            train_data_file (str): Path to the training data `.bin` file.
            val_data_file (str): Path to the validation data `.bin` file.
            batch_size (int): The number of sequences in each training batch.
            learning_rate (float): The learning rate for the optimizer.
            max_iters (int): The total number of training iterations to run.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the __init__ method.")

    @abstractmethod
    def train(self) -> None:
        """
        Executes the main training loop.

        CONTRACT:
        - Must load training and validation data from the specified files.
        - Must iterate for `max_iters`, performing the following in each step:
            1. Sample a batch of data.
            2. Perform a forward and backward pass to compute gradients.
            3. Update the model's weights using an optimizer.
        - Must periodically evaluate the model's loss on the validation set.
        - Must save at least one model checkpoint to disk upon completion or
          at regular intervals.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement the train method.")