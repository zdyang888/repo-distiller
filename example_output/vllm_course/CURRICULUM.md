# Curriculum: Understanding vLLM: Fast LLM Inference and Serving

Source: https://github.com/vllm-project/vllm

## Overview

The vLLM system receives user requests through its LLMEngine, which then dispatches them to an EngineCore via an EngineCoreClient. The EngineCore orchestrates the entire inference process, utilizing a Scheduler to manage incoming requests, prioritize them, and allocate KV cache blocks using PagedAttention. An Executor component then takes the scheduled batches and runs the actual model inference, potentially across multiple devices, while interacting with the KV cache and returning outputs to the Scheduler, which are then processed back to the user.

## Notebooks

### 01. Interacting with vLLM: The User API

Learn how users interact with vLLM by sending requests and receiving responses, understanding the LLMEngine's role as the primary interface to the system.

**Learning objectives:**
- By the end, students will be able to initiate and manage LLM inference requests through a high-level API.
- Students will understand the basic lifecycle of a request from submission to output retrieval.

**Exercise:** Implement a simplified `RequestManager` class that queues requests, assigns unique IDs, and simulates marking them as 'completed' after a mock processing step.

### 02. The Heart of vLLM: Request Orchestration with EngineCore

Understand how the EngineCore orchestrates the entire inference lifecycle, dispatching requests and coordinating between scheduling and execution components.

**Learning objectives:**
- By the end, students will be able to describe the role of EngineCore as the central coordinating entity.
- Students will understand how EngineCore manages the flow of requests between different internal components.

**Exercise:** Design a mock `EngineCore` that receives requests from a mock `LLMEngine`, logs its 'dispatch' decisions to mock `Scheduler` and `Executor` interfaces, and tracks request states.

### 03. Efficient Request Handling: The Scheduler

Dive into how the Scheduler component manages the queue of incoming requests, prioritizes them, and prepares batches for model execution, considering resource constraints.

**Learning objectives:**
- By the end, students will be able to explain how requests are queued, prioritized, and selected for inference.
- Students will understand the scheduler's role in managing inference throughput and latency.

**Exercise:** Implement a basic `FirstComeFirstServedScheduler` that maintains a queue of incoming requests, selects a subset to form an 'execution batch', and updates request states (e.g., pending, running, completed).

### 04. Model Execution: The Executor

Explore how the Executor component is responsible for loading the LLM model and performing the actual forward pass on scheduled batches of requests.

**Learning objectives:**
- By the end, students will be able to describe the executor's responsibility in loading and running LLM models.
- Students will understand the interface between the scheduler and the low-level model inference operations.

**Exercise:** Create a mock `SimpleModelExecutor` class that takes a batch of 'token IDs' and simulates generating new token IDs, along with managing simplified KV cache updates, returning the generated tokens and 'completion' status.

### 05. Memory Efficiency: PagedAttention KV Cache

Understand the advanced PagedAttention technique for managing KV cache memory, enabling higher throughput and efficient resource utilization by treating KV cache as a paged memory system.

**Learning objectives:**
- By the end, students will be able to explain the core principles of PagedAttention for KV cache management.
- Students will understand how block allocation and deallocation contribute to memory efficiency and throughput.

**Exercise:** Implement a simplified `BlockAllocator` that manages a fixed pool of 'KV cache pages', allocating them to simulated requests, handling block extensions for growing sequences, and correctly deallocating blocks upon request completion.

## Capstone Project

**mini-vLLM Inference Server**

Students will build a simplified, end-to-end LLM inference server that processes user requests, schedules them based on availability, executes a mock model for token generation, and manages a basic KV cache using PagedAttention principles, synthesizing all core vLLM concepts.

---

<!-- Edit the JSON below to skip notebooks (set skip: true) -->

```json
{
  "title": "Understanding vLLM: Fast LLM Inference and Serving",
  "mental_model": "The vLLM system receives user requests through its LLMEngine, which then dispatches them to an EngineCore via an EngineCoreClient. The EngineCore orchestrates the entire inference process, utilizing a Scheduler to manage incoming requests, prioritize them, and allocate KV cache blocks using PagedAttention. An Executor component then takes the scheduled batches and runs the actual model inference, potentially across multiple devices, while interacting with the KV cache and returning outputs to the Scheduler, which are then processed back to the user.",
  "concepts": [
    {
      "name": "LLMEngine (User API)",
      "description": "The primary user-facing API for interacting with vLLM, providing methods to add requests, retrieve outputs, and manage the lifecycle of inference jobs.",
      "complexity": "basic"
    },
    {
      "name": "EngineCore",
      "description": "The central orchestration component of vLLM, responsible for the inner loop of the inference process, integrating the Scheduler and Executor.",
      "complexity": "intermediate"
    },
    {
      "name": "Scheduler",
      "description": "Manages the lifecycle of inference requests, including queuing, prioritizing, scheduling for execution, and allocating KV cache blocks using PagedAttention.",
      "complexity": "intermediate"
    },
    {
      "name": "Executor",
      "description": "An abstract component responsible for executing the LLM model, managing model loading, distributed inference, and low-level interaction with the KV cache.",
      "complexity": "intermediate"
    },
    {
      "name": "PagedAttention",
      "description": "A key memory optimization technique treating KV cache as a paged memory system, allowing flexible allocation and sharing of KV cache blocks among requests for higher throughput.",
      "complexity": "advanced"
    }
  ],
  "notebooks": [
    {
      "id": "01",
      "title": "Interacting with vLLM: The User API",
      "concept": "LLMEngine (User API)",
      "description": "Learn how users interact with vLLM by sending requests and receiving responses, understanding the LLMEngine's role as the primary interface to the system.",
      "prerequisites": [],
      "key_source_files": [
        "vllm/v1/engine/llm_engine.py"
      ],
      "key_symbols": [
        "LLMEngine",
        "LLMEngine.add_request",
        "LLMEngine.step"
      ],
      "learning_objectives": [
        "By the end, students will be able to initiate and manage LLM inference requests through a high-level API.",
        "Students will understand the basic lifecycle of a request from submission to output retrieval."
      ],
      "exercise_description": "Implement a simplified `RequestManager` class that queues requests, assigns unique IDs, and simulates marking them as 'completed' after a mock processing step.",
      "visualization_idea": "Graphviz diagram showing the data flow of a user request entering the LLMEngine and moving into an internal queue."
    },
    {
      "id": "02",
      "title": "The Heart of vLLM: Request Orchestration with EngineCore",
      "concept": "EngineCore",
      "description": "Understand how the EngineCore orchestrates the entire inference lifecycle, dispatching requests and coordinating between scheduling and execution components.",
      "prerequisites": [
        "01"
      ],
      "key_source_files": [
        "vllm/v1/engine/core.py",
        "vllm/v1/engine/core_client.py"
      ],
      "key_symbols": [
        "EngineCore",
        "EngineCore.step",
        "EngineCoreClient"
      ],
      "learning_objectives": [
        "By the end, students will be able to describe the role of EngineCore as the central coordinating entity.",
        "Students will understand how EngineCore manages the flow of requests between different internal components."
      ],
      "exercise_description": "Design a mock `EngineCore` that receives requests from a mock `LLMEngine`, logs its 'dispatch' decisions to mock `Scheduler` and `Executor` interfaces, and tracks request states.",
      "visualization_idea": "Graphviz block diagram illustrating the interaction between LLMEngine, EngineCore, Scheduler, and Executor, highlighting data flow paths."
    },
    {
      "id": "03",
      "title": "Efficient Request Handling: The Scheduler",
      "concept": "Scheduler",
      "description": "Dive into how the Scheduler component manages the queue of incoming requests, prioritizes them, and prepares batches for model execution, considering resource constraints.",
      "prerequisites": [
        "01",
        "02"
      ],
      "key_source_files": [
        "vllm/v1/core/sched/interface.py"
      ],
      "key_symbols": [
        "Scheduler",
        "Scheduler.schedule"
      ],
      "learning_objectives": [
        "By the end, students will be able to explain how requests are queued, prioritized, and selected for inference.",
        "Students will understand the scheduler's role in managing inference throughput and latency."
      ],
      "exercise_description": "Implement a basic `FirstComeFirstServedScheduler` that maintains a queue of incoming requests, selects a subset to form an 'execution batch', and updates request states (e.g., pending, running, completed).",
      "visualization_idea": "Graphviz state machine diagram showing the transitions of an inference request through different states within the scheduler (e.g., 'queued', 'scheduled', 'processing', 'completed')."
    },
    {
      "id": "04",
      "title": "Model Execution: The Executor",
      "concept": "Executor",
      "description": "Explore how the Executor component is responsible for loading the LLM model and performing the actual forward pass on scheduled batches of requests.",
      "prerequisites": [
        "01",
        "02",
        "03"
      ],
      "key_source_files": [
        "vllm/v1/executor/abstract.py"
      ],
      "key_symbols": [
        "ExecutorBase",
        "ExecutorBase.execute_model"
      ],
      "learning_objectives": [
        "By the end, students will be able to describe the executor's responsibility in loading and running LLM models.",
        "Students will understand the interface between the scheduler and the low-level model inference operations."
      ],
      "exercise_description": "Create a mock `SimpleModelExecutor` class that takes a batch of 'token IDs' and simulates generating new token IDs, along with managing simplified KV cache updates, returning the generated tokens and 'completion' status.",
      "visualization_idea": "High-level Graphviz data flow diagram showing a batch of requests entering the Executor, interacting with a 'mock LLM' and 'mock KV cache', and producing output tokens."
    },
    {
      "id": "05",
      "title": "Memory Efficiency: PagedAttention KV Cache",
      "concept": "PagedAttention",
      "description": "Understand the advanced PagedAttention technique for managing KV cache memory, enabling higher throughput and efficient resource utilization by treating KV cache as a paged memory system.",
      "prerequisites": [
        "01",
        "02",
        "03",
        "04"
      ],
      "key_source_files": [
        "vllm/v1/worker/cache_engine.py",
        "vllm/v1/core/kv_cache_utils.py",
        "vllm/v1/core/sched/interface.py"
      ],
      "key_symbols": [
        "CacheEngine",
        "BlockTable"
      ],
      "learning_objectives": [
        "By the end, students will be able to explain the core principles of PagedAttention for KV cache management.",
        "Students will understand how block allocation and deallocation contribute to memory efficiency and throughput."
      ],
      "exercise_description": "Implement a simplified `BlockAllocator` that manages a fixed pool of 'KV cache pages', allocating them to simulated requests, handling block extensions for growing sequences, and correctly deallocating blocks upon request completion.",
      "visualization_idea": "Matplotlib diagram showing a grid representing physical KV cache blocks, with different colors indicating allocation to various 'simulated requests' and highlighting fragmented vs. contiguous allocation scenarios."
    }
  ],
  "capstone": {
    "title": "mini-vLLM Inference Server",
    "description": "Students will build a simplified, end-to-end LLM inference server that processes user requests, schedules them based on availability, executes a mock model for token generation, and manages a basic KV cache using PagedAttention principles, synthesizing all core vLLM concepts.",
    "estimated_hours": 4,
    "modules": [
      {
        "name": "MockLLMEngine",
        "description": "A simplified user-facing API for adding and managing inference requests.",
        "depends_on": [
          "MiniEngineCore"
        ],
        "interface_sketch": "class MockLLMEngine:\n    def __init__(self, core_client):\n        ...\n    def add_request(self, prompt: str, request_id: str):\n        ...\n    def step(self):\n        ...\n    def get_outputs(self) -> dict: # {request_id: output_text}\n        ...",
        "test_behaviors": [
          "Given a prompt, add_request should add it to an internal queue and send to core_client.",
          "After step() is called, completed requests should be available via get_outputs() with correct results.",
          "Adding a request with a duplicate ID should raise ValueError."
        ]
      },
      {
        "name": "MiniEngineCore",
        "description": "The central orchestrator, coordinating request flow between the scheduler and executor.",
        "depends_on": [
          "BasicScheduler",
          "DummyExecutor"
        ],
        "interface_sketch": "class MiniEngineCore:\n    def __init__(self, scheduler, executor):\n        ...\n    def process_new_requests(self, new_requests: list):\n        # Takes new requests from MockLLMEngine\n        ...\n    def execute_step(self) -> dict: # Returns completed outputs\n        ...",
        "test_behaviors": [
          "Given new requests, process_new_requests should pass them to the scheduler.",
          "execute_step should call the scheduler's schedule method and the executor's execute method with a batch.",
          "If the scheduler returns no pending requests, execute_step should not call the executor and return an empty dict."
        ]
      },
      {
        "name": "BasicScheduler",
        "description": "Manages request queues, prioritizes them, and interacts with the KV block manager for memory allocation.",
        "depends_on": [
          "SimpleKVBlockManager"
        ],
        "interface_sketch": "class BasicScheduler:\n    def __init__(self, kv_block_manager):\n        ...\n    def add_request(self, request_id: str, prompt: str, max_output_len: int):\n        ...\n    def schedule(self) -> list: # Returns a list of (request_id, block_table) for execution\n        ...\n    def update_request_state(self, request_id: str, new_token: str, is_finished: bool):\n        ...",
        "test_behaviors": [
          "Adding a request should allocate initial KV cache blocks via SimpleKVBlockManager.",
          "schedule() should return pending requests with valid block tables when blocks are available.",
          "Updating a request state as finished should deallocate its KV cache blocks.",
          "schedule() should prioritize requests based on arrival time (FCFS principle)."
        ]
      },
      {
        "name": "DummyExecutor",
        "description": "Simulates the LLM model execution, generating mock tokens for a given prompt.",
        "depends_on": [],
        "interface_sketch": "class DummyExecutor:\n    def __init__(self, mock_responses: dict):\n        # mock_responses = {'prompt': 'response'}\n        ...\n    def execute_batch(self, scheduled_requests: list) -> list: # Returns list of (request_id, new_token, is_finished)\n        ...",
        "test_behaviors": [
          "Given a request with a known prompt, execute_batch should return the expected next token from its mock response.",
          "If the request reaches its max tokens or the mock response ends, it should be marked as finished.",
          "Executing an unknown prompt should return a default or error token and mark as finished."
        ]
      },
      {
        "name": "SimpleKVBlockManager",
        "description": "Manages the allocation and deallocation of fixed-size KV cache blocks, mimicking PagedAttention.",
        "depends_on": [],
        "interface_sketch": "class SimpleKVBlockManager:\n    def __init__(self, num_total_blocks: int):\n        ...\n    def allocate(self, num_blocks: int) -> list: # Returns list of block IDs\n        ...\n    def free(self, block_ids: list):\n        ...\n    def get_available_blocks(self) -> int:\n        ...",
        "test_behaviors": [
          "allocate(N) should return N unique block IDs if N blocks are available.",
          "allocate(N) should raise an error if N blocks are not available (out of memory).",
          "free(block_ids) should mark blocks as available for subsequent reuse.",
          "After allocating and freeing blocks, the number of available blocks should return to its initial state."
        ]
      }
    ],
    "integration_test": {
      "description": "Test the entire mini-vLLM system by submitting multiple concurrent requests, ensuring correct sequence generation, proper KV cache utilization, and graceful handling of memory limits.",
      "setup_code": "mock_model_responses = {\n    \"hello world\": \"hello world, how are you?\",\n    \"tell me a joke\": \"Why don't scientists trust atoms? Because they make up everything!\",\n    \"short query\": \"short answer\"\n}\nkv_cache_size = 5 # Small cache to test allocation/deallocation\nengine = MockLLMEngine(MiniEngineCore(BasicScheduler(SimpleKVBlockManager(kv_cache_size)), DummyExecutor(mock_model_responses)))\nengine.add_request(\"hello world\", \"req1\")\nengine.add_request(\"tell me a joke\", \"req2\")\nengine.add_request(\"short query\", \"req3\")",
      "success_metric": "All submitted requests successfully complete, their outputs match the mock responses, and the system does not crash due to out-of-memory errors (even with limited `kv_cache_size`).",
      "expected_output_check": "Verify that `engine.get_outputs()` contains 'req1': 'hello world, how are you?', 'req2': 'Why don't scientists trust atoms? Because they make up everything!', and 'req3': 'short answer'. Verify that no errors related to KV cache block allocation are raised during execution."
    }
  }
}
```