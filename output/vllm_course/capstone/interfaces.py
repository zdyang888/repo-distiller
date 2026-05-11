"""
Module: interfaces
Description:
    This file defines the abstract base classes (ABCs) for the core components
    of a simplified LLM inference engine, designed for a capstone project.
    The architecture is inspired by systems like vLLM and demonstrates key
    concepts such as continuous batching, paged attention-style memory
    management, and request scheduling.

    The primary components are:
    - Request: A data class representing a single inference request.
    - IRequestQueue: Manages incoming requests before they are scheduled.
    - IKVCacheSimulator: Simulates the allocation and management of KV cache
      memory blocks.
    - IModelRunner: Represents the LLM, responsible for executing a forward
      pass on a batch of requests.
    - IBatchScheduler: The core orchestrator that creates batches from running
      and pending requests, manages request states, and interacts with the
      KV cache simulator.
    - IMiniLLMEngine: The top-level facade that integrates all components and
      provides the end-user API for generating text.
"""

from abc import ABC, abstractmethod
from collections import deque
from typing import Iterator, Any


# --------------------------------------------------------------------------
# Data Structures and Concrete Utility Classes
# --------------------------------------------------------------------------

class Request:
    """
    A data class representing a single inference request.

    This class holds all state associated with a request as it moves through
    the system, from pending to running to completion.
    """

    def __init__(self, request_id: str, prompt_tokens: list[int], max_tokens: int):
        """
        Initializes a new inference Request.

        Args:
            request_id: A unique identifier for the request.
            prompt_tokens: The list of token IDs for the input prompt.
            max_tokens: The maximum number of tokens to generate for this request.
        """
        self.request_id: str = request_id
        self.prompt_tokens: list[int] = prompt_tokens
        self.output_tokens: list[int] = []
        self.status: str = "PENDING"  # PENDING, RUNNING, COMPLETED, ERROR
        self.max_tokens: int = max_tokens
        self.kv_cache_block_ids: list[int] = []  # Physically allocated blocks


class SimpleTokenizer:
    """A simple, concrete tokenizer for demonstration purposes."""

    def encode(self, text: str) -> list[int]:
        """Encodes a string into a list of token IDs (ASCII values)."""
        # Simple char-to-int for alphabetic characters and spaces
        return [ord(c) for c in text if c.isalpha() or c.isspace()]

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        return "".join([chr(t) for t in tokens])


# --------------------------------------------------------------------------
# Abstract Base Classes (Interfaces)
# --------------------------------------------------------------------------

class IRequestQueue(ABC):
    """
    Manages incoming inference requests, holding them until they can be processed.

    This component is the entry point for new requests into the system. It is
    responsible for tokenizing the raw prompt text and creating a structured
    Request object for the scheduler to consume.
    """

    @abstractmethod
    def add_request(self, prompt: str, max_tokens: int) -> Request:
        """
        Tokenizes a prompt and adds it as a new request to the queue.

        The contract is to create a `Request` object with a unique ID,
        tokenize the prompt, store the request, and set its initial status
        to 'PENDING'.

        Args:
            prompt: The input text to be processed by the LLM.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The newly created Request object.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_pending_requests(self) -> list[Request]:
        """
        Retrieves all requests that are currently in the 'PENDING' state.

        The contract is to return a list of requests that have been added but
        not yet scheduled for execution by the BatchScheduler.

        Returns:
            A list of Request objects with 'PENDING' status.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_request(self, request_id: str) -> Request | None:
        """
        Retrieves a specific request by its ID.

        Args:
            request_id: The unique identifier of the request to retrieve.

        Returns:
            The Request object if found, otherwise None.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def remove_request(self, request_id: str) -> None:
        """
        Removes a completed or failed request from the internal tracking.

        This method is typically called by the scheduler after a request has
        finished processing to clean up resources.

        Args:
            request_id: The unique identifier of the request to remove.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class IKVCacheSimulator(ABC):
    """
    Manages the allocation and deallocation of fixed-size KV cache blocks.

    This class simulates the core memory management functionality of systems
    like PagedAttention, providing a pool of memory blocks that can be
    dynamically allocated to and freed from running requests.
    """

    @abstractmethod
    def allocate_blocks(self, request_id: str, num_blocks: int) -> list[int]:
        """
        Allocates a specified number of KV cache blocks for a given request.

        This method must find `num_blocks` free blocks, mark them as allocated
        to `request_id`, and return their unique identifiers. It must also
        decrease the count of available free blocks.

        Args:
            request_id: The unique identifier of the request requiring blocks.
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of unique integer block IDs that have been allocated.

        Raises:
            ValueError: If not enough free blocks are available to fulfill
                        the allocation request.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def free_blocks_for_request(self, request_id: str) -> None:
        """
        Returns all blocks allocated to a request back to the free pool.

        This is called when a request is completed, aborted, or encounters an
        error to reclaim its memory resources.

        Args:
            request_id: The ID of the request whose blocks should be freed.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_num_free_blocks(self) -> int:
        """
        Returns the total number of currently available KV cache blocks.

        Returns:
            An integer count of free blocks.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class IModelRunner(ABC):
    """
    Simulates an LLM's forward pass for a batch of requests.

    This interface represents the model execution component. Its core
    responsibility is to take a batch of requests and produce the next
    token for each of them.
    """

    @abstractmethod
    def run_forward_pass(self, batch_requests: list[Request]) -> dict[str, int]:
        """
        Executes a single forward pass for a batch of requests.

        The contract is to take a list of `Request` objects and return a
        dictionary mapping each request's ID to its newly generated next
        token ID.

        Args:
            batch_requests: A list of `Request` objects to process.

        Returns:
            A dictionary where keys are request_ids and values are the
            generated next token ID for that request.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class IBatchScheduler(ABC):
    """
    Prioritizes, batches requests, and orchestrates KV cache management.

    This is the central controller of the inference engine. It decides which
    requests to run in the next model pass (continuous batching), allocates
    and deallocates KV cache blocks as needed, and updates the status of
    requests as they progress.
    """

    @abstractmethod
    def schedule(self, pending_requests: list[Request], max_batch_size: int) -> list[Request]:
        """
        Creates a new batch for the next model forward pass.

        The contract is to:
        1. Prioritize currently running requests to include them in the new batch
           (this is the core of continuous batching).
        2. Attempt to add new 'PENDING' requests to the batch if there is space
           and sufficient KV cache blocks are available.
        3. For new requests, allocate initial KV cache blocks and update their
           status to 'RUNNING'.

        Args:
            pending_requests: A list of requests waiting to be scheduled.
            max_batch_size: The maximum number of requests allowed in the batch.

        Returns:
            A list of `Request` objects that constitute the batch for the
            next forward pass.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def update_request_status(self, request: Request, new_status: str) -> None:
        """
        Updates the status of a request and performs associated cleanup.

        The contract is to change the request's status. If the new status is
        'COMPLETED' or 'ERROR', it must also free the KV cache blocks
        associated with that request.

        Args:
            request: The `Request` object to update.
            new_status: The new status string (e.g., "COMPLETED", "ERROR").
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def add_token_to_request(self, request: Request, new_token_id: int) -> None:
        """
        Appends a newly generated token to a request's output.

        The contract is to add the `new_token_id` to the request's output list
        and check if additional KV cache blocks are needed to store the state
        for the new token. If more blocks are needed, it must try to allocate
        them. If allocation fails, the request's status should be set to 'ERROR'.

        Args:
            request: The `Request` object to which the token should be added.
            new_token_id: The ID of the token that was just generated.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class IMiniLLMEngine(ABC):
    """
    The top-level class that integrates all components for end-to-end inference.

    This interface provides the public API for the LLM inference engine. It
    hides the complexity of scheduling, batching, and memory management,
    offering a simple method to generate text from a prompt.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 16) -> Iterator[str]:
        """
        Generates text for a given prompt in a streaming fashion.

        The contract is to:
        1. Create a new request and add it to the request queue.
        2. Repeatedly execute the engine's internal step (scheduling, model
           execution, state update).
        3. Yield newly generated text chunks as they become available for the
           submitted request.
        4. Continue until the request is completed (reaches `max_tokens`) or
           an error occurs.

        Args:
            prompt: The input text to generate from.
            max_tokens: The maximum number of new tokens to generate.

        Yields:
            Strings representing chunks of newly generated text.

        Raises:
            RuntimeError: If the request fails during processing.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")