Here is a detailed capstone project README for students.

---

# mini-vLLM: A Simplified LLM Inference Server

Welcome to your capstone project! By completing this project, you will have built a simplified, end-to-end LLM inference server from scratch. You'll gain a deep, practical understanding of the core concepts that make modern LLM serving systems like vLLM incredibly fast and efficient, including PagedAttention-style memory management, continuous batching, and request scheduling.

This project is designed to be a hands-on experience, simulating the architecture of a high-throughput inference system in about 4 hours. You will implement the key components that work together to serve multiple users concurrently and efficiently.

## 🎯 Project Goal

The objective is to build a `MiniLLMEngine` that can process multiple, concurrent text generation requests. This engine will not use a real LLM but will simulate its key behaviors, allowing you to focus on the systems architecture that enables high performance.

## 🏛️ System Architecture

Your system will be composed of five main components that work in a coordinated loop. The `MiniLLMEngine` orchestrates this loop, processing requests from a queue, scheduling them into batches, executing a "model," and managing memory.

Here is a diagram of the data flow:

```ascii
  User Code                 +-------------------------------------------------------------+
(e.g., generate())          | MiniLLMEngine (_step() loop)                                |
       |                    |                                                             |
       v                    |    +----------------+   1. Add   +--------------+           |
.----------------------.    |    |   User Prompt  | ---------> | RequestQueue |           |
|  Streaming Iterator  |    |    '----------------'            +--------------+           |
'----------------------'    |                                       ^   | 2. Get Pending |
       ^ (Yields tokens)    |                                       |   v                |
       | 6. Process         |    +----------------+           +----------------+           |
       +--------------------|----| MockModelRunner| <---------| BatchScheduler |           |
                            |    +----------------+  5. Run   +----------------+           |
                            |      ^       |       Forward        ^   | 3. Schedule Batch |
                            |      |       | Pass                 |   |                   |
                            |      +-------+                      |   v                   |
                            |     4. Batch of Requests       +--------------------+       |
                            |                                | KVCacheSimulator   |       |
                            |                                | (Paged KV Cache)   |       |
                            |                                +--------------------+       |
                            |                                                             |
                            +-------------------------------------------------------------+
```

## 🧩 Modules to Implement

You will implement five Python classes in `implementation.py`. Each class has a specific role in the system.

### 1. `RequestQueue`

*   **Responsibility**: Manages incoming inference requests. It accepts user prompts, tokenizes them, and stores them in a queue until they are ready to be processed by the scheduler.
*   **Interface**:
    ```python
    import collections
    from typing import List, Dict, Any

    class Request:
        def __init__(self, request_id: str, prompt_tokens: List[int], max_tokens: int):
            self.request_id = request_id
            self.prompt_tokens = prompt_tokens
            self.output_tokens: List[int] = []
            self.status: str = "PENDING" # PENDING, RUNNING, COMPLETED, ERROR
            self.max_tokens = max_tokens
            self.kv_cache_block_ids: List[int] = [] # Physically allocated blocks

    class RequestQueue:
        def __init__(self, tokenizer):
            self._queue = collections.deque() # Stores request_ids
            self._requests: Dict[str, Request] = {}
            self.tokenizer = tokenizer

        def add_request(self, prompt: str, max_tokens: int = 16) -> Request:
            # ... implementation needed ...

        def get_pending_requests(self) -> List[Request]:
            # ... implementation needed ...

        def get_request(self, request_id: str) -> Request:
            # ... implementation needed ...

        def remove_request(self, request_id: str):
            # ... implementation needed ...
    ```
*   **Expected Behavior**:
    *   Given a prompt, `add_request` should tokenize it and store a new `Request` object with 'PENDING' status.
    *   Calling `get_pending_requests` should return only requests with 'PENDING' status.

### 2. `KVCacheSimulator`

*   **Responsibility**: Simulates the core memory management concept of vLLM's PagedAttention. It manages a pool of fixed-size KV cache "blocks" and handles their allocation and deallocation for different requests.
*   **Interface**:
    ```python
    from typing import Dict, List

    class KVCacheSimulator:
        def __init__(self, total_blocks: int, block_size: int):
            self.total_blocks = total_blocks
            self.block_size = block_size
            self.free_blocks = list(range(total_blocks))
            self.allocated_blocks: Dict[str, List[int]] = {} # request_id -> list of block_ids

        def allocate_blocks(self, request_id: str, num_blocks: int) -> List[int]:
            # ... implementation needed ...

        def free_blocks_for_request(self, request_id: str):
            # ... implementation needed ...

        def get_num_free_blocks(self) -> int:
            # ... implementation needed ...
    ```
*   **Expected Behavior**:
    *   Calling `allocate_blocks` should return unique block IDs and decrease the count of free blocks.
    *   Calling `free_blocks_for_request` should return allocated blocks to the free pool and increase the count of free blocks.
    *   Attempting to allocate more blocks than available should raise a `ValueError`.

### 3. `MockModelRunner`

*   **Responsibility**: Simulates the forward pass of an LLM. Instead of performing complex GPU computations, it accepts a batch of requests and returns a deterministically generated "next token" for each one. This allows us to focus on the system's logic without needing a real model.
*   **Interface**:
    ```python
    from typing import List, Dict, Any
    # Assuming Request class is available

    class MockModelRunner:
        def __init__(self, vocab_size: int = 32000):
            self.vocab_size = vocab_size
            self.mock_logits_counter = 0

        def run_forward_pass(self, batch_requests: List[Any]) -> Dict[str, int]: # Any assumes Request
            # ... implementation needed ...
    ```
*   **Expected Behavior**:
    *   Given a batch of requests, `run_forward_pass` should return a dictionary of next token IDs, one for each request.
    *   The generated token IDs should be within the mock vocabulary range (e.g., >0 and <vocab_size).

### 4. `BatchScheduler`

*   **Responsibility**: This is the brain of the operation. It decides which requests to run in the next model iteration. It implements **continuous batching** by including already-running requests alongside new, pending requests. It also orchestrates KV cache allocation/deallocation via the `KVCacheSimulator`.
*   **Interface**:
    ```python
    from typing import List, Dict, Any
    # Assuming Request class and KVCacheSimulator are available

    class BatchScheduler:
        def __init__(self, kv_cache_simulator: Any): # Any assumes KVCacheSimulator
            self.kv_cache_simulator = kv_cache_simulator
            self.running_requests: Dict[str, Any] = {} # request_id -> Request

        def schedule(self, pending_requests: List[Any], max_batch_size: int) -> List[Any]: # Any assumes Request
            # ... implementation needed ...

        def update_request_status(self, request: Any, new_status: str):
            # ... implementation needed ...

        def add_token_to_request(self, request: Any, new_token_id: int):
            # ... implementation needed ...
    ```
*   **Expected Behavior**:
    *   Given pending requests and available KV cache blocks, `schedule` should add new requests to the batch and allocate initial blocks for them.
    *   If a request is `RUNNING`, it should be prioritized and included in the next batch (continuous batching).
    *   When a request completes, `update_request_status` should free its allocated KV cache blocks.
    *   If `add_token_to_request` requires more blocks and none are available, the request's status should change to 'ERROR'.

### 5. `MiniLLMEngine`

*   **Responsibility**: The top-level class that integrates all other components. It exposes the public `generate` API, runs the main `_step` loop to drive the inference process, and provides the streaming output back to the user.
*   **Interface**:
    ```python
    import time
    from typing import Iterator, List, Dict, Any
    from collections import deque
    # Assuming other classes are available

    class MiniLLMEngine:
        def __init__(self, total_kv_blocks: int = 100, kv_block_size: int = 4, max_batch_size: int = 4):
            # ... initialization ...

        def _step(self):
            # ... implementation needed ...

        def generate(self, prompt: str, max_tokens: int = 16) -> Iterator[str]:
            # ... implementation needed ...
    ```
*   **Expected Behavior**:
    *   Calling `generate` with a prompt should return an iterator that yields generated token chunks over time.
    *   The `generate` method should eventually yield all `max_tokens` for a request unless an error occurs.
    *   Multiple concurrent calls to `generate` (e.g., from different threads) should be handled correctly, demonstrating the power of continuous batching.

## 🤝 How the Pieces Fit Together

The magic of the system happens in the `MiniLLMEngine._step()` method, which runs continuously as long as there are active requests.

1.  **Request Arrival**: A user calls `engine.generate()`. A new `Request` is created and added to the `RequestQueue`.
2.  **Scheduling a Batch**: In each `_step()`, the `BatchScheduler` is called. It first adds all `RUNNING` requests to the next batch. Then, it tries to fill the rest of the batch with `PENDING` requests from the `RequestQueue`, but only if it can successfully allocate initial KV cache blocks for them from the `KVCacheSimulator`.
3.  **Model Execution**: The engine sends the scheduled batch to the `MockModelRunner`, which returns a new token for every request in the batch.
4.  **State Update**: The engine processes the results. For each request:
    *   It calls `scheduler.add_token_to_request()`. This appends the new token and allocates more KV cache blocks if the current ones are full. If allocation fails, the request is marked as an `ERROR`.
    *   The new token is added to a stream buffer for the user.
    *   It checks if the request has reached its `max_tokens`. If so, it's marked `COMPLETED`, and the scheduler frees all its KV cache blocks.
5.  **Streaming Output**: The `generate()` method's loop yields any new tokens from the stream buffer back to the user, then pauses briefly, allowing the `_step()` loop (driven by other concurrent calls) to continue making progress.

## ✅ Success Criteria

Your implementation is successful when all automated tests pass. The tests are designed to validate each component in isolation and then the entire system working together.

*   **Unit Tests**: Each module (`RequestQueue`, `KVCacheSimulator`, etc.) has its own set of tests in `test_capstone.py`.
*   **Integration Test**: A final test, `test_integration_concurrent_requests`, simulates a real-world scenario. It uses multiple threads to submit 8 concurrent requests to your engine. To pass, your engine must:
    *   Process all requests to completion without deadlocking or crashing.
    *   Correctly generate exactly `max_tokens_per_request` for each successful request.
    *   Efficiently manage KV cache memory to handle the concurrent load.

## 🚀 Suggested Implementation Order

We recommend implementing the components in order of their dependencies to allow for incremental testing.

1.  **`RequestQueue`**: A simple, standalone data structure.
2.  **`KVCacheSimulator`**: Another standalone component, critical for memory management.
3.  **`MockModelRunner`**: A very simple, dependency-free simulator.
4.  **`BatchScheduler`**: This is the most complex piece of logic. It depends on the three components above. Take your time here to get the scheduling and memory allocation logic right.
5.  **`MiniLLMEngine`**: The final step is to wire everything together in the engine's `_step` and `generate` methods.

## 💻 Getting Started

1.  Navigate to the project directory:
    ```bash
    cd capstone/
    ```

2.  Open `implementation.py`. This is the only file you need to edit. It contains the class skeletons for all the modules described above.

3.  Fill in the logic for each class, following the interface sketches and expected behaviors.

4.  As you complete each module, you can run the tests to check your work. To run all tests, use `pytest`:
    ```bash
    pytest test_capstone.py -v
    ```

    You can also run tests for a specific class, for example:
    ```bash
    pytest test_capstone.py -k TestRequestQueue -v
    ```

### ⚠️ A Note on Tests

The provided tests are your guide to a correct implementation. However, please be aware of the following notice:

> These tests could not be fully validated against a reference implementation. Some tests may have issues. If a failing test appears to be wrong (not your implementation), please file an issue.

Good luck