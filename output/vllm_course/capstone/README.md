# Capstone Project: mini-vLLM Inference Server

By completing this project, you will have built a simplified, end-to-end LLM inference server from scratch. You will gain a deep, practical understanding of the core concepts that make modern LLM serving systems like vLLM incredibly fast and efficient, including request scheduling, batching, and the revolutionary PagedAttention mechanism for managing KV cache memory.

This project synthesizes all core vLLM concepts into a manageable, 4-hour challenge. You will implement the central components that handle request processing, memory management, and simulated model execution.

## System Architecture

The mini-vLLM server is composed of several distinct modules that work together to process inference requests. The user interacts with the high-level `MockLLMEngine`, which orchestrates the entire process through the `MiniEngineCore`.

Here is a high-level diagram of the architecture:

```
       User Request (prompt, request_id)
             |
             v
   +-------------------+
   |   MockLLMEngine   | (User-facing API)
   +-------------------+
      |      ^
 add_request | get_outputs
      |      |
      v      |
   +-------------------+
   |  MiniEngineCore   | (Orchestrator)
   +-------------------+
      |      ^      |
 process_new |      | execute_step
      |      |      |
      v      |      v
   +-------------------+     schedule()    +-----------------+
   |  BasicScheduler   |------------------>|  DummyExecutor  |
   | (Manages Queues)  |<------------------| (Mock Model)    |
   +-------------------+   update_state    +-----------------+
      |           ^
 allocate() | free()
      |           |
      v           |
   +--------------------------+
   |  SimpleKVBlockManager    | (PagedAttention Memory)
   +--------------------------+
```

## Module Breakdown

You will implement five key modules. Each module has a specific responsibility and a clear interface.

---

### 1. `SimpleKVBlockManager`

*   **Responsibility:** Manages the allocation and deallocation of fixed-size KV cache blocks, mimicking vLLM's PagedAttention. This is the lowest-level memory manager.
*   **Interface:**
    ```python
    class SimpleKVBlockManager:
        def __init__(self, num_total_blocks: int):
            ...
        def allocate(self, num_blocks: int) -> list: # Returns list of block IDs
            ...
        def free(self, block_ids: list):
            ...
        def get_available_blocks(self) -> int:
            ...
    ```
*   **Expected Behavior:**
    *   `allocate(N)` should return N unique block IDs if N blocks are available.
    *   `allocate(N)` should raise an error if N blocks are not available (out of memory).
    *   `free(block_ids)` should mark blocks as available for subsequent reuse.
    *   After allocating and freeing blocks, the number of available blocks should return to its initial state.

---

### 2. `DummyExecutor`

*   **Responsibility:** Simulates the LLM model execution. Instead of running a real GPU model, it looks up a prompt in a dictionary and "generates" the pre-written response one token at a time.
*   **Interface:**
    ```python
    class DummyExecutor:
        def __init__(self, mock_responses: dict):
            # mock_responses = {'prompt': 'response'}
            ...
        def execute_batch(self, scheduled_requests: list) -> list: # Returns list of (request_id, new_token, is_finished)
            ...
    ```
*   **Expected Behavior:**
    *   Given a request with a known prompt, `execute_batch` should return the expected next token from its mock response.
    *   If the request reaches its max tokens or the mock response ends, it should be marked as finished.
    *   Executing an unknown prompt should return a default or error token and mark as finished.

---

### 3. `BasicScheduler`

*   **Responsibility:** The brain of the operation. It manages request queues (`waiting`, `running`), decides which requests to run next based on a First-Come, First-Served (FCFS) policy, and interacts with the `SimpleKVBlockManager` to allocate memory for them.
*   **Interface:**
    ```python
    class BasicScheduler:
        def __init__(self, kv_block_manager):
            ...
        def add_request(self, request_id: str, prompt: str, max_output_len: int):
            ...
        def schedule(self) -> list: # Returns a list of (request_id, block_table) for execution
            ...
        def update_request_state(self, request_id: str, new_token: str, is_finished: bool):
            ...
    ```
*   **Expected Behavior:**
    *   Adding a request should allocate initial KV cache blocks via `SimpleKVBlockManager`.
    *   `schedule()` should return pending requests with valid block tables when blocks are available.
    *   Updating a request state as finished should deallocate its KV cache blocks.
    *   `schedule()` should prioritize requests based on arrival time (FCFS principle).

---

### 4. `MiniEngineCore`

*   **Responsibility:** The central orchestrator. It acts as the glue between the scheduler and the executor, managing the main execution loop.
*   **Interface:**
    ```python
    class MiniEngineCore:
        def __init__(self, scheduler, executor):
            ...
        def process_new_requests(self, new_requests: list):
            # Takes new requests from MockLLMEngine
            ...
        def execute_step(self) -> dict: # Returns completed outputs
            ...
    ```
*   **Expected Behavior:**
    *   Given new requests, `process_new_requests` should pass them to the scheduler.
    *   `execute_step` should call the scheduler's `schedule` method and the executor's `execute` method with a batch.
    *   If the scheduler returns no pending requests, `execute_step` should not call the executor and should return an empty dict.

---

### 5. `MockLLMEngine`

*   **Responsibility:** A simplified user-facing API. This is the entry point for submitting new inference requests and retrieving completed results. It hides the complexity of the internal core loop.
*   **Interface:**
    ```python
    class MockLLMEngine:
        def __init__(self, core_client):
            ...
        def add_request(self, prompt: str, request_id: str):
            ...
        def step(self):
            ...
        def get_outputs(self) -> dict: # {request_id: output_text}
            ...
    ```
*   **Expected Behavior:**
    *   Given a prompt, `add_request` should add it to an internal queue and send it to the `core_client`.
    *   After `step()` is called, completed requests should be available via `get_outputs()` with correct results.
    *   Adding a request with a duplicate ID should raise `ValueError`.

## How Modules Connect: The Lifecycle of a Request

1.  A user calls `MockLLMEngine.add_request()`. The request is queued.
2.  The user calls `MockLLMEngine.step()`.
3.  The `MockLLMEngine` sends all new requests to `MiniEngineCore.process_new_requests()`.
4.  The `MiniEngineCore` passes these new requests to the `BasicScheduler.add_request()`.
5.  The `BasicScheduler` attempts to allocate initial KV cache blocks for the prompt from the `SimpleKVBlockManager`. If it fails, the request waits.
6.  The `MiniEngineCore` then calls `BasicScheduler.schedule()` to get a batch of requests that are ready to run (i.e., have memory allocated).
7.  This batch is sent to `DummyExecutor.execute_batch()`.
8.  The `DummyExecutor` "generates" the next token for each request in the batch and returns the results, including whether each request is now finished.
9.  The `MiniEngineCore` passes these results to `BasicScheduler.update_request_state()`.
10. If a request is finished, the `BasicScheduler` calls `SimpleKVBlockManager.free()` to release its KV cache blocks, making memory available for other waiting requests.
11. The `MiniEngineCore` collects the full text of any completed requests and returns them.
12. Finally, the `MockLLMEngine` stores these completed outputs, which the user can retrieve with `get_outputs()`.

## Success Criteria and Testing

Your implementation is considered correct and complete when all `pytest` tests pass. These tests cover each module in isolation and then verify the entire system with an integration test.

### Final Integration Test

The final test will configure the entire system to ensure it can handle multiple concurrent requests with a limited memory budget, simulating a real-world scenario.

*   **Description:** Test the entire mini-vLLM system by submitting multiple concurrent requests, ensuring correct sequence generation, proper KV cache utilization, and graceful handling of memory limits.
*   **Setup:**
    ```python
    mock_model_responses = {
        "hello world": "hello world, how are you?",
        "tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
        "short query": "short answer"
    }
    kv_cache_size = 5 # Small cache to test allocation/deallocation
    engine = MockLLMEngine(MiniEngineCore(BasicScheduler(SimpleKVBlockManager(kv_cache_size)), DummyExecutor(mock_model_responses)))
    engine.add_request("hello world", "req1")
    engine.add_request("tell me a joke", "req2")
    engine.add_request("short query", "req3")
    ```
*   **Success Metric:** All submitted requests successfully complete, their outputs match the mock responses, and the system does not crash due to out-of-memory errors (even with the limited `kv_cache_size`).
*   **Expected Output Check:** Verify that `engine.get_outputs()` contains `'req1': 'hello world, how are you?'`, `'req2': 'Why don't scientists trust atoms? Because they make up everything!'`, and `'req3': 'short answer'`. Verify that no errors related to KV cache block allocation are raised during execution.

## Suggested Implementation Order

We strongly recommend implementing the modules in a "bottom-up" order based on their dependencies. This allows you to build and potentially test each layer before moving to the next.

1.  **`SimpleKVBlockManager`** (No dependencies)
2.  **`DummyExecutor`** (No dependencies)
3.  **`BasicScheduler`** (Depends on `SimpleKVBlockManager`)
4.  **`MiniEngineCore`** (Depends on `BasicScheduler` and `DummyExecutor`)
5.  **`MockLLMEngine`** (Depends on `MiniEngineCore`)

## Getting Started

All your work should be done in the `implementation.py` file. The tests in `test_capstone.py` will import your classes from that file.

1.  Navigate to the project directory:
    ```bash
    cd capstone/
    ```

2.  Open `implementation.py` in your editor and begin implementing the classes as described above.

3.  Run the tests from your terminal to check your progress:
    ```bash
    pytest test_capstone.py -v
    ```

Good luck!

---

### ⚠️ Validation Notice

These tests could not be fully validated against a reference implementation.
Some tests may have issues. If a failing test appears to be wrong (not your implementation), please file an issue.

Validation history:
- Attempt 1: ============================= test session starts ==============================
collecting ... collected 19 items

test_capstone.py::test_SimpleKVBlockManager_allocation PASSED            [  5%]
test_capstone.py::test_SimpleKVBlockManager_out_of_memory PASSED         [ 10%]
test_capstone.py::test_SimpleKVBlockManager_freeing_blocks PASSED        [ 15%]
test_capstone.py::test_SimpleKVBlockManager_reuse_freed_blocks PASSED    [ 21%]
test_capstone.py::test_DummyExecutor_generates_next_token FAILED
- Attempt 2: ============================= test session starts ==============================
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
______________________ ERROR collecting test_capstone.py _______________________
/Users/tsetungy/miniconda3/lib/python3.12/site-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/Users/tsetungy/miniconda3/lib/python3.12/site-packages/_pytest/pathlib.py:587: in import_p
- Attempt 3: ============================= test session starts ==============================
collecting ... collected 0 items / 1 error

==================================== ERRORS ====================================
______________________ ERROR collecting test_capstone.py _______________________
/Users/tsetungy/miniconda3/lib/python3.12/site-packages/_pytest/python.py:507: in importtestmodule
    mod = import_path(
/Users/tsetungy/miniconda3/lib/python3.12/site-packages/_pytest/pathlib.py:587: in import_p