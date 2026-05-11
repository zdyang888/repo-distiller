"""
Interfaces for a simplified LLM Inference Engine Capstone Project.

This file defines the abstract base classes (ABCs) that form the contract
for each component of the inference engine. The architecture is designed to be
modular, allowing for different implementations of each component to be
swapped in.

The overall data flow is as follows:
1.  A user-facing client (`MockLLMEngine`) accepts new inference requests.
2.  The `MockLLMEngine` forwards these requests to the central orchestrator,
    the `MiniEngineCore`.
3.  The `MiniEngineCore` passes new requests to the `BasicScheduler`.
4.  The `BasicScheduler` manages request queues. It uses a
    `SimpleKVBlockManager` to check for memory availability (KV cache blocks)
    and decide which requests to run in the next iteration.
5.  The `MiniEngineCore` takes the scheduled batch of requests and sends them
    to the `DummyExecutor` for processing.
6.  The `DummyExecutor` simulates model execution, generating the next token
    for each request in the batch.
7.  The `MiniEngineCore` receives the results and updates the `BasicScheduler`
    with the new state of each request (e.g., new token, finished status).
8.  This cycle repeats, with the `MockLLMEngine`'s `step()` method driving
    each iteration of the engine's execution loop.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple


class SimpleKVBlockManager(ABC):
    """
    Manages the allocation and deallocation of fixed-size KV cache blocks.

    This class abstracts the memory management of the KV cache, mimicking
    systems like PagedAttention. It keeps track of a pool of memory blocks
    and provides methods to allocate, free, and query the number of
    available blocks.
    """

    @abstractmethod
    def __init__(self, num_total_blocks: int):
        """
        Initializes the block manager with a fixed total number of blocks.

        Args:
            num_total_blocks: The total number of KV cache blocks available in
                              the pool.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def allocate(self, num_blocks: int) -> list[int]:
        """
        Allocates a specified number of blocks from the free pool.

        CONTRACT:
        - Must return a list of unique integer block IDs.
        - The length of the returned list must be equal to `num_blocks`.
        - If not enough blocks are available to satisfy the request, this
          method must raise an exception (e.g., ValueError or a custom
          OutOfMemoryError).
        - The allocated blocks must be marked as 'in-use' and should not be
          available for subsequent allocations until freed.

        Args:
            num_blocks: The number of blocks to allocate.

        Returns:
            A list of integers representing the unique IDs of the allocated blocks.

        Raises:
            ValueError: If `num_blocks` cannot be satisfied due to insufficient
                        available blocks.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def free(self, block_ids: list[int]) -> None:
        """
        Returns a list of block IDs to the free pool.

        CONTRACT:
        - Must mark the blocks corresponding to the given `block_ids` as
          'available'.
        - After this call, the freed blocks must be available for subsequent
          `allocate` calls.
        - Attempting to free a block that is already free or invalid may
          result in an error or be ignored, depending on the implementation's
          strictness.

        Args:
            block_ids: A list of block IDs to deallocate.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_available_blocks(self) -> int:
        """
        Returns the current number of available (free) blocks.

        CONTRACT:
        - Must return a non-negative integer representing the count of blocks
          that can currently be allocated.

        Returns:
            The number of free blocks.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class DummyExecutor(ABC):
    """
    Simulates the LLM model execution for a batch of requests.

    This class takes a batch of scheduled requests and generates the next token
    for each, based on a predefined mapping of prompts to responses. It is
    a stand-in for a real model runner (e.g., a PyTorch model).
    """

    @abstractmethod
    def __init__(self, mock_responses: dict[str, str]):
        """
        Initializes the executor with a dictionary of mock prompt-response pairs.

        Args:
            mock_responses: A dictionary where keys are prompts and values are
                            the complete expected output strings.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def execute_batch(self, scheduled_requests: list[Any]) -> list[tuple[str, str, bool]]:
        """
        Processes a batch of requests and produces the next token for each.

        CONTRACT:
        - Must iterate through `scheduled_requests` and determine the next token
          for each one.
        - The `is_finished` flag in the return tuple must be True if the
          generated token is the last one for that request (based on the mock
          response) or if a generation limit is reached.
        - If a request's prompt is not found in the mock responses, it should
          handle it gracefully (e.g., return a default token and mark as
          finished).

        Args:
            scheduled_requests: A list of requests to be processed. The exact
                                structure of each item is determined by the
                                scheduler, but it must contain enough information
                                to identify the request and its current state
                                (e.g., request_id, prompt, tokens generated so far).

        Returns:
            A list of tuples, where each tuple contains:
            - request_id (str): The ID of the request.
            - new_token (str): The newly generated token.
            - is_finished (bool): True if the request has completed generation.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class BasicScheduler(ABC):
    """
    Manages request queues, scheduling, and KV cache block allocation.

    This class is responsible for maintaining queues of waiting, running, and
    finished requests. In each scheduling step, it decides which requests can
    be processed based on memory availability from the KVBlockManager,
    implementing a scheduling policy (e.g., First-Come, First-Served).
    """

    @abstractmethod
    def __init__(self, kv_block_manager: SimpleKVBlockManager):
        """
        Initializes the scheduler with a KV block manager.

        Args:
            kv_block_manager: An instance of a SimpleKVBlockManager (or its
                              subclass) to manage KV cache memory.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def add_request(self, request_id: str, prompt: str, max_output_len: int) -> None:
        """
        Adds a new request to the scheduler's queue.

        CONTRACT:
        - Must add the new request to a 'waiting' or 'pending' queue.
        - May attempt to pre-allocate initial KV cache blocks for the prompt
          tokens. If allocation fails, the request should be kept in the
          waiting queue until memory is available.

        Args:
            request_id: A unique identifier for the request.
            prompt: The input text for the language model.
            max_output_len: The maximum number of tokens to generate for this request.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def schedule(self) -> list[Any]:
        """
        Determines which requests to run in the next execution step.

        CONTRACT:
        - Must implement a scheduling logic (e.g., FCFS).
        - Must check the `SimpleKVBlockManager` for available blocks before
          scheduling a new request or continuing a running one that needs more
          memory.
        - Must return a list of requests that are ready for execution by the
          DummyExecutor. Each item in the list should contain at least the
          request_id and its associated KV block table.
        - If no requests can be scheduled (e.g., all are waiting for memory),
          it must return an empty list.

        Returns:
            A list of scheduled requests ready for the executor. The exact
            structure of each item is an implementation detail.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def update_request_state(self, request_id: str, new_token: str, is_finished: bool) -> None:
        """
        Updates the state of a request after an execution step.

        CONTRACT:
        - Must append `new_token` to the specified request's generated output.
        - If `is_finished` is True, the request must be moved to a 'finished'
          state, and all its associated KV cache blocks must be freed using the
          `SimpleKVBlockManager`.
        - If `is_finished` is False, the scheduler may need to allocate a new
          KV block for the new token if the current one is full.

        Args:
            request_id: The ID of the request to update.
            new_token: The token that was just generated.
            is_finished: A boolean flag indicating if the request is now complete.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class MiniEngineCore(ABC):
    """
    The central orchestrator of the inference engine.

    This class coordinates the flow of requests between the scheduler and the
    executor. It exposes a simple `execute_step` method that drives one
    iteration of the scheduling and execution cycle.
    """

    @abstractmethod
    def __init__(self, scheduler: BasicScheduler, executor: DummyExecutor):
        """
        Initializes the core engine with a scheduler and an executor.

        Args:
            scheduler: An instance of a BasicScheduler (or subclass).
            executor: An instance of a DummyExecutor (or subclass).
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def process_new_requests(self, new_requests: list[Any]) -> None:
        """
        Accepts new requests and forwards them to the scheduler.

        CONTRACT:
        - Must iterate through `new_requests` and call the scheduler's
          `add_request` method for each one.

        Args:
            new_requests: A list of new requests to be added to the system.
                          The structure of each request item should be consistent
                          with what the scheduler's `add_request` expects.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def execute_step(self) -> dict[str, str]:
        """
        Executes one full step of the inference process.

        CONTRACT:
        - Must first call the scheduler's `schedule()` method to get a batch
          of requests to run.
        - If the batch is not empty, it must pass it to the executor's
          `execute_batch()` method.
        - It must then take the results from the executor and use them to
          update the scheduler's state via `update_request_state()`.
        - Must return a dictionary of any requests that completed during this
          step, mapping request_id to its full generated output text.
        - If the scheduler returns an empty batch, this method should not call
          the executor and should return an empty dictionary.

        Returns:
            A dictionary of completed request outputs from this step.
            Format: {request_id: full_output_string}.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")


class MockLLMEngine(ABC):
    """
    A simplified user-facing API for managing inference requests.

    This class provides a high-level interface for users to add prompts and
    drive the engine's execution. It hides the internal complexity of the
    core, scheduler, and executor.
    """

    @abstractmethod
    def __init__(self, core_client: MiniEngineCore):
        """
        Initializes the user-facing engine with a core client.

        Args:
            core_client: An instance of the MiniEngineCore that this API will
                         interact with.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def add_request(self, prompt: str, request_id: str) -> None:
        """
        Adds a new inference request to the engine.

        CONTRACT:
        - Must store the new request in an internal queue to be processed
          in the next `step()` call.
        - Must raise a ValueError if a request with the same `request_id`
          already exists or is being processed.

        Args:
            prompt: The input text for the language model.
            request_id: A unique string to identify this request.

        Raises:
            ValueError: If the `request_id` is a duplicate.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def step(self) -> None:
        """
        Advances the engine by one execution step.

        CONTRACT:
        - Must first pass any newly added requests (from `add_request`) to the
          `MiniEngineCore` via its `process_new_requests` method.
        - Must then call the `MiniEngineCore`'s `execute_step` method.
        - Must collect and store any completed outputs returned by `execute_step`.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def get_outputs(self) -> dict[str, str]:
        """
        Retrieves the outputs of all completed requests.

        CONTRACT:
        - Must return a dictionary containing all requests that have finished
          processing since the engine was started.
        - The dictionary should map each `request_id` to its complete generated
          output string.

        Returns:
            A dictionary of completed request outputs.
        """
        # TODO: Implement this
        raise NotImplementedError("Subclasses must implement this method.")