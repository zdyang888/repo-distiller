# Curriculum: Understanding vLLM

Source: https://github.com/vllm-project/vllm

## Overview

The system processes incoming requests through an `LLMEngine`, which uses an `InputProcessor` to prepare them and an `OutputProcessor` to format results. The core of the system is the `EngineCore`, which coordinates a `Scheduler` and an `Executor`. The `Scheduler` manages the lifecycle of requests and their KV cache blocks (via PagedAttention), while the `Executor` is responsible for running the model on the underlying hardware, performing the actual forward passes.

## Notebooks

### 01. Request Input & Output Processing

Students will build simplified versions of an `InputProcessor` to tokenize prompts and an `OutputProcessor` to detokenize generated tokens and manage streaming output.

**Learning objectives:**
- By the end, students will be able to understand the lifecycle of a request's input and output within an LLM serving system.
- Students will understand why tokenization and detokenization are crucial for LLM interaction.

**Exercise:** Students will implement a `SimpleInputProcessor` to tokenize a string into integer IDs and a `SimpleOutputProcessor` to detokenize a list of IDs back to a string, handling partial outputs for streaming.

### 02. PagedAttention: Efficient KV Cache Management

Students will simulate the core mechanics of PagedAttention, managing fixed-size KV cache blocks for multiple sequences, and understanding how it reduces memory fragmentation to improve throughput.

**Learning objectives:**
- By the end, students will be able to understand the concept of PagedAttention and its benefits for LLM inference memory efficiency.
- Students will understand how to implement a simplified KV cache block allocation and mapping system.

**Exercise:** Students will implement a `SimulatedKVCacheManager` that allocates fixed-size `KVBlock`s and maps logical sequence token positions to physical block-offset pairs, mimicking PagedAttention.

### 03. The LLM Model Executor

Students will build a mock `Executor` that simulates running a model's forward pass, accepting batches of token IDs and returning mock logits, focusing on the interface between the scheduler and the model.

**Learning objectives:**
- By the end, students will be able to understand the role of the `Executor` in abstracting model execution details.
- Students will understand how to simulate a batched LLM forward pass on a computational device.

**Exercise:** Students will implement a `MockExecutor` that takes a batch of `Request` objects (with input IDs and mock KV cache mappings) and returns a batch of `MockOutput` objects (with generated token IDs), simulating a single model step.

### 04. Request Scheduling and Continuous Batching

Students will implement a simplified `Scheduler` that manages multiple incoming requests, prioritizes them, and constructs batches for the `Executor` using continuous batching logic, interacting with the KV cache manager.

**Learning objectives:**
- By the end, students will be able to understand continuous batching and its role in maximizing LLM throughput.
- Students will understand how to design a scheduling algorithm that manages sequence states and interacts with a KV cache system.

**Exercise:** Students will implement a `SimpleScheduler` that queues incoming `requests`, selects a batch for the `MockExecutor` based on available KV cache blocks (from `SimulatedKVCacheManager`), and updates sequence states through continuous batching.

### 05. Orchestrating the Inference Engine Core

Students will integrate the `Scheduler`, `Executor`, and `Input/Output Processors` into a basic `EngineCore`, demonstrating the central coordination of inference requests from intake to output.

**Learning objectives:**
- By the end, students will be able to understand how the core components of an LLM serving system interact.
- Students will understand how to build an orchestrator that manages the flow from request intake to output generation.

**Exercise:** Students will implement a `MiniEngineCore` that initializes and coordinates the `SimpleScheduler`, `MockExecutor`, `SimpleInputProcessor`, and `SimpleOutputProcessor` to process a single end-to-end request.

### 06. The User-Facing LLM Engine API

Students will create a high-level `LLMEngine` that provides a user-friendly, streaming interface for submitting prompts and receiving responses, utilizing the `EngineCore`.

**Learning objectives:**
- By the end, students will be able to understand the public API and user interaction patterns of an LLM serving system.
- Students will understand how to implement an asynchronous or streaming interface for LLM inference.

**Exercise:** Students will implement a `UserLLMEngine` class that wraps the `MiniEngineCore`, offering a `generate(prompt: str, ...)` method that returns an iterator for streaming output.

## Capstone Project

**mini-vLLM**

Students will build a simplified, end-to-end LLM inference server that leverages the key concepts of vLLM: request processing, efficient KV cache management, continuous batching, and model execution. This capstone will solidify their understanding of how these components integrate to achieve high-throughput LLM serving.

---

<!-- Edit the JSON below to skip notebooks (set skip: true) -->

```json
{
  "title": "Understanding vLLM",
  "mental_model": "The system processes incoming requests through an `LLMEngine`, which uses an `InputProcessor` to prepare them and an `OutputProcessor` to format results. The core of the system is the `EngineCore`, which coordinates a `Scheduler` and an `Executor`. The `Scheduler` manages the lifecycle of requests and their KV cache blocks (via PagedAttention), while the `Executor` is responsible for running the model on the underlying hardware, performing the actual forward passes.",
  "concepts": [
    {
      "name": "LLMEngine",
      "description": "The public-facing entry point for interacting with vLLM, responsible for receiving user requests, delegating input/output processing, and coordinating with the internal `EngineCore` for model execution and scheduling.",
      "complexity": "basic"
    },
    {
      "name": "EngineCore",
      "description": "The central orchestrator of the vLLM system, responsible for initializing the `ModelExecutor` and `Scheduler`, managing the overall lifecycle of inference, and handling communication with external clients (via `EngineCoreClient`).",
      "complexity": "intermediate"
    },
    {
      "name": "Scheduler",
      "description": "Manages the efficient scheduling of LLM inference requests, including techniques like continuous batching and PagedAttention for KV cache management. It determines which requests to process in each model forward pass and handles their memory allocation and state transitions.",
      "complexity": "intermediate"
    },
    {
      "name": "Executor",
      "description": "Responsible for executing the LLM model on the computational devices (GPUs/CPUs). It manages the workers (potentially distributed), initializes model weights and KV caches, and performs the actual forward passes based on the `Scheduler`'s output.",
      "complexity": "intermediate"
    },
    {
      "name": "PagedAttention",
      "description": "A key memory management technique inspired by virtual memory and paging in operating systems. It efficiently manages the KV cache by breaking it into fixed-size blocks, allowing for non-contiguous memory allocation and reducing memory fragmentation, leading to higher throughput.",
      "complexity": "advanced"
    },
    {
      "name": "InputProcessor / OutputProcessor",
      "description": "The `InputProcessor` handles the parsing, validation, and tokenization of incoming user prompts, converting them into an internal format (`EngineCoreRequest`). The `OutputProcessor` takes the model's raw outputs (`EngineCoreOutputs`) and formats them into user-friendly `RequestOutput` objects, including detokenization and managing streaming.",
      "complexity": "basic"
    }
  ],
  "notebooks": [
    {
      "id": "01",
      "title": "Request Input & Output Processing",
      "concept": "InputProcessor / OutputProcessor",
      "description": "Students will build simplified versions of an `InputProcessor` to tokenize prompts and an `OutputProcessor` to detokenize generated tokens and manage streaming output.",
      "prerequisites": [],
      "key_source_files": [
        "vllm/v1/engine/input_processor.py",
        "vllm/v1/engine/output_processor.py"
      ],
      "key_symbols": [
        "InputProcessor",
        "OutputProcessor"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand the lifecycle of a request's input and output within an LLM serving system.",
        "Students will understand why tokenization and detokenization are crucial for LLM interaction."
      ],
      "exercise_description": "Students will implement a `SimpleInputProcessor` to tokenize a string into integer IDs and a `SimpleOutputProcessor` to detokenize a list of IDs back to a string, handling partial outputs for streaming.",
      "visualization_idea": "Animate a prompt string being tokenized into integer IDs, then IDs being incrementally detokenized into an output string, showing the stream."
    },
    {
      "id": "02",
      "title": "PagedAttention: Efficient KV Cache Management",
      "concept": "PagedAttention",
      "description": "Students will simulate the core mechanics of PagedAttention, managing fixed-size KV cache blocks for multiple sequences, and understanding how it reduces memory fragmentation to improve throughput.",
      "prerequisites": [],
      "key_source_files": [
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/kv_cache_utils.py"
      ],
      "key_symbols": [
        "KVCacheManager"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand the concept of PagedAttention and its benefits for LLM inference memory efficiency.",
        "Students will understand how to implement a simplified KV cache block allocation and mapping system."
      ],
      "exercise_description": "Students will implement a `SimulatedKVCacheManager` that allocates fixed-size `KVBlock`s and maps logical sequence token positions to physical block-offset pairs, mimicking PagedAttention.",
      "visualization_idea": "Visualize sequences requesting KV cache space, and `KVBlock`s being allocated and mapped in non-contiguous physical memory, highlighting fragmentation reduction."
    },
    {
      "id": "03",
      "title": "The LLM Model Executor",
      "concept": "Executor",
      "description": "Students will build a mock `Executor` that simulates running a model's forward pass, accepting batches of token IDs and returning mock logits, focusing on the interface between the scheduler and the model.",
      "prerequisites": [],
      "key_source_files": [
        "vllm/v1/executor/abstract.py",
        "vllm/v1/executor/uniproc_executor.py"
      ],
      "key_symbols": [
        "ExecutorBase",
        "UniprocessExecutor"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand the role of the `Executor` in abstracting model execution details.",
        "Students will understand how to simulate a batched LLM forward pass on a computational device."
      ],
      "exercise_description": "Students will implement a `MockExecutor` that takes a batch of `Request` objects (with input IDs and mock KV cache mappings) and returns a batch of `MockOutput` objects (with generated token IDs), simulating a single model step.",
      "visualization_idea": "Show a batch of input sequences entering the `Executor`, a forward pass occurring, and output tokens being generated and returned for each sequence."
    },
    {
      "id": "04",
      "title": "Request Scheduling and Continuous Batching",
      "concept": "Scheduler",
      "description": "Students will implement a simplified `Scheduler` that manages multiple incoming requests, prioritizes them, and constructs batches for the `Executor` using continuous batching logic, interacting with the KV cache manager.",
      "prerequisites": [
        "PagedAttention",
        "Executor"
      ],
      "key_source_files": [
        "vllm/v1/core/sched/interface.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/kv_cache_manager.py"
      ],
      "key_symbols": [
        "LlmScheduler",
        "KVCacheManager"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand continuous batching and its role in maximizing LLM throughput.",
        "Students will understand how to design a scheduling algorithm that manages sequence states and interacts with a KV cache system."
      ],
      "exercise_description": "Students will implement a `SimpleScheduler` that queues incoming `requests`, selects a batch for the `MockExecutor` based on available KV cache blocks (from `SimulatedKVCacheManager`), and updates sequence states through continuous batching.",
      "visualization_idea": "Animate multiple requests arriving, being queued, then batched together by the scheduler before being sent to the executor. Show KV cache blocks being dynamically allocated/deallocated."
    },
    {
      "id": "05",
      "title": "Orchestrating the Inference Engine Core",
      "concept": "EngineCore",
      "description": "Students will integrate the `Scheduler`, `Executor`, and `Input/Output Processors` into a basic `EngineCore`, demonstrating the central coordination of inference requests from intake to output.",
      "prerequisites": [
        "Scheduler",
        "Executor",
        "InputProcessor / OutputProcessor"
      ],
      "key_source_files": [
        "vllm/v1/engine/core.py"
      ],
      "key_symbols": [
        "EngineCore"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand how the core components of an LLM serving system interact.",
        "Students will understand how to build an orchestrator that manages the flow from request intake to output generation."
      ],
      "exercise_description": "Students will implement a `MiniEngineCore` that initializes and coordinates the `SimpleScheduler`, `MockExecutor`, `SimpleInputProcessor`, and `SimpleOutputProcessor` to process a single end-to-end request.",
      "visualization_idea": "Show the `MiniEngineCore` as a central hub, with requests flowing from InputProcessor, through Scheduler and Executor, and out through OutputProcessor in a loop."
    },
    {
      "id": "06",
      "title": "The User-Facing LLM Engine API",
      "concept": "LLMEngine",
      "description": "Students will create a high-level `LLMEngine` that provides a user-friendly, streaming interface for submitting prompts and receiving responses, utilizing the `EngineCore`.",
      "prerequisites": [
        "EngineCore"
      ],
      "key_source_files": [
        "vllm/v1/engine/llm_engine.py"
      ],
      "key_symbols": [
        "LLMEngine"
      ],
      "learning_objectives": [
        "By the end, students will be able to understand the public API and user interaction patterns of an LLM serving system.",
        "Students will understand how to implement an asynchronous or streaming interface for LLM inference."
      ],
      "exercise_description": "Students will implement a `UserLLMEngine` class that wraps the `MiniEngineCore`, offering a `generate(prompt: str, ...)` method that returns an iterator for streaming output.",
      "visualization_idea": "Illustrate how a user request interacts with the `UserLLMEngine` and then flows down into the `MiniEngineCore` and its components, with generated results streaming back to the user."
    }
  ],
  "capstone": {
    "title": "mini-vLLM",
    "description": "Students will build a simplified, end-to-end LLM inference server that leverages the key concepts of vLLM: request processing, efficient KV cache management, continuous batching, and model execution. This capstone will solidify their understanding of how these components integrate to achieve high-throughput LLM serving.",
    "estimated_hours": 4,
    "modules": [
      {
        "name": "RequestQueue",
        "description": "Manages incoming inference requests, holding them until the scheduler can process them. Also handles basic input processing (tokenization).",
        "depends_on": [],
        "interface_sketch": "import collections\nfrom typing import List, Dict, Any\n\nclass Request:\n    def __init__(self, request_id: str, prompt_tokens: List[int], max_tokens: int):\n        self.request_id = request_id\n        self.prompt_tokens = prompt_tokens\n        self.output_tokens: List[int] = []\n        self.status: str = \"PENDING\" # PENDING, RUNNING, COMPLETED, ERROR\n        self.max_tokens = max_tokens\n        self.kv_cache_block_ids: List[int] = [] # Physically allocated blocks\n\nclass RequestQueue:\n    def __init__(self, tokenizer):\n        self._queue = collections.deque() # Stores request_ids\n        self._requests: Dict[str, Request] = {}\n        self.tokenizer = tokenizer\n\n    def add_request(self, prompt: str, max_tokens: int = 16) -> Request:\n        request_id = f\"req-{len(self._requests)}\"\n        prompt_tokens = self.tokenizer.encode(prompt)\n        request = Request(request_id, prompt_tokens, max_tokens)\n        self._queue.append(request_id)\n        self._requests[request_id] = request\n        return request\n\n    def get_pending_requests(self) -> List[Request]:\n        return [self._requests[req_id] for req_id in self._queue if self._requests[req_id].status == \"PENDING\"]\n\n    def get_request(self, request_id: str) -> Request:\n        return self._requests[request_id]\n\n    def remove_request(self, request_id: str):\n        if request_id in self._requests:\n            # In a real system, the scheduler would manage removal from its queues.\n            # For this capstone, we simply mark completed and assume eventually removed.\n            # For now, let's just remove it if status is completed.\n            # This simplifies the get_pending_requests for capstone.\n            if self._requests[request_id].status in (\"COMPLETED\", \"ERROR\"):\n                # Remove from deque if present (might be redundant if scheduler removed)\n                try: self._queue.remove(request_id) # O(N) but simple for capstone\n                except ValueError: pass\n                del self._requests[request_id]\n",
        "test_behaviors": [
          "Given a prompt, `add_request` should tokenize it and store a new `Request` object with 'PENDING' status.",
          "Calling `get_pending_requests` should return only requests with 'PENDING' status."
        ]
      },
      {
        "name": "KVCacheSimulator",
        "description": "Manages the allocation and deallocation of fixed-size KV cache blocks for sequences, simulating PagedAttention's memory management.",
        "depends_on": [],
        "interface_sketch": "from typing import Dict, List\n\nclass KVCacheSimulator:\n    def __init__(self, total_blocks: int, block_size: int):\n        self.total_blocks = total_blocks\n        self.block_size = block_size\n        self.free_blocks = list(range(total_blocks))\n        self.allocated_blocks: Dict[str, List[int]] = {} # request_id -> list of block_ids\n\n    def allocate_blocks(self, request_id: str, num_blocks: int) -> List[int]:\n        if len(self.free_blocks) < num_blocks:\n            raise ValueError(\"Not enough free KV cache blocks\")\n        \n        allocated = [self.free_blocks.pop(0) for _ in range(num_blocks)]\n        self.allocated_blocks[request_id] = allocated\n        return allocated\n\n    def free_blocks_for_request(self, request_id: str):\n        if request_id in self.allocated_blocks:\n            for block_id in self.allocated_blocks[request_id]:\n                self.free_blocks.append(block_id)\n            del self.allocated_blocks[request_id]\n            self.free_blocks.sort() # Keep sorted for deterministic behavior\n\n    def get_num_free_blocks(self) -> int:\n        return len(self.free_blocks)\n",
        "test_behaviors": [
          "Calling `allocate_blocks` should return unique block IDs and decrease the count of free blocks.",
          "Calling `free_blocks_for_request` should return allocated blocks to the free pool and increase the count of free blocks.",
          "Attempting to allocate more blocks than available should raise a `ValueError`."
        ]
      },
      {
        "name": "MockModelRunner",
        "description": "Simulates an LLM's forward pass, accepting a batch of inputs and generating next token IDs for each request in the batch.",
        "depends_on": [],
        "interface_sketch": "from typing import List, Dict, Any\n# Assuming Request class is available (e.g., from RequestQueue definition)\n\nclass MockModelRunner:\n    def __init__(self, vocab_size: int = 32000):\n        self.vocab_size = vocab_size\n        self.mock_logits_counter = 0\n\n    def run_forward_pass(self, batch_requests: List[Any]) -> Dict[str, int]: # Any assumes Request\n        # Simulates generating a single next token for each request in the batch\n        generated_tokens = {}\n        for req in batch_requests:\n            # Mock token generation: just increment a counter within vocab size\n            next_token_id = (self.mock_logits_counter % (self.vocab_size - 1)) + 1\n            self.mock_logits_counter += 1\n            generated_tokens[req.request_id] = next_token_id\n        return generated_tokens\n",
        "test_behaviors": [
          "Given a batch of requests, `run_forward_pass` should return a dictionary of next token IDs, one for each request.",
          "The generated token IDs should be within the mock vocabulary range (e.g., >0 and <vocab_size)."
        ]
      },
      {
        "name": "BatchScheduler",
        "description": "Prioritizes and batches requests for the model, manages their state (running, completed), and orchestrates KV cache block allocation and deallocation.",
        "depends_on": [
          "RequestQueue",
          "KVCacheSimulator",
          "MockModelRunner"
        ],
        "interface_sketch": "from typing import List, Dict, Any\n# Assuming Request class and KVCacheSimulator are available\n\nclass BatchScheduler:\n    def __init__(self, kv_cache_simulator: Any): # Any assumes KVCacheSimulator\n        self.kv_cache_simulator = kv_cache_simulator\n        self.running_requests: Dict[str, Any] = {} # request_id -> Request\n\n    def schedule(self, pending_requests: List[Any], max_batch_size: int) -> List[Any]: # Any assumes Request\n        batch: List[Any] = []\n        \n        # Add existing running requests first (continuous batching)\n        for req_id in list(self.running_requests.keys()): # Iterate over copy as dict might change\n            req = self.running_requests[req_id]\n            if req.status == \"RUNNING\":\n                batch.append(req)\n            \n        # Try to add new requests from pending queue\n        for req in pending_requests:\n            if req.status == \"PENDING\" and len(batch) < max_batch_size:\n                # Simplified block allocation: allocate for prompt + 1 initial token\n                blocks_needed_for_prompt = (len(req.prompt_tokens) + self.kv_cache_simulator.block_size - 1) // self.kv_cache_simulator.block_size\n                initial_blocks_estimate = blocks_needed_for_prompt + 1 # For prompt and first generated token\n\n                if self.kv_cache_simulator.get_num_free_blocks() >= initial_blocks_estimate:\n                    try:\n                        req.kv_cache_block_ids = self.kv_cache_simulator.allocate_blocks(req.request_id, initial_blocks_estimate)\n                        req.status = \"RUNNING\"\n                        batch.append(req)\n                        self.running_requests[req.request_id] = req\n                    except ValueError:\n                        pass # Not enough blocks, skip for now\n        return batch\n\n    def update_request_status(self, request: Any, new_status: str):\n        request.status = new_status\n        if new_status in (\"COMPLETED\", \"ERROR\"):\n            self.kv_cache_simulator.free_blocks_for_request(request.request_id)\n            if request.request_id in self.running_requests:\n                del self.running_requests[request.request_id]\n\n    def add_token_to_request(self, request: Any, new_token_id: int):\n        request.output_tokens.append(new_token_id)\n        # Check if more blocks are needed for generated tokens\n        current_total_tokens = len(request.prompt_tokens) + len(request.output_tokens)\n        blocks_needed = (current_total_tokens + self.kv_cache_simulator.block_size - 1) // self.kv_cache_simulator.block_size\n        \n        current_blocks_allocated = len(request.kv_cache_block_ids)\n        if blocks_needed > current_blocks_allocated:\n            num_new_blocks = blocks_needed - current_blocks_allocated\n            try:\n                new_blocks = self.kv_cache_simulator.allocate_blocks(request.request_id, num_new_blocks)\n                request.kv_cache_block_ids.extend(new_blocks)\n            except ValueError:\n                # Out of blocks during generation. Mark as error and free resources.\n                self.update_request_status(request, \"ERROR\")\n",
        "test_behaviors": [
          "Given pending requests and available KV cache blocks, `schedule` should add new requests to the batch and allocate initial blocks for them.",
          "If a request is `RUNNING`, it should be prioritized and included in the next batch (continuous batching).",
          "When a request completes, `update_request_status` should free its allocated KV cache blocks.",
          "If `add_token_to_request` requires more blocks and none are available, the request's status should change to 'ERROR'."
        ]
      },
      {
        "name": "MiniLLMEngine",
        "description": "The top-level class that integrates all components (`RequestQueue`, `KVCacheSimulator`, `MockModelRunner`, `BatchScheduler`) to provide an end-to-end LLM inference API with streaming output.",
        "depends_on": [
          "RequestQueue",
          "KVCacheSimulator",
          "MockModelRunner",
          "BatchScheduler"
        ],
        "interface_sketch": "import time\nfrom typing import Iterator, List, Dict, Any\nfrom collections import deque\n\n# SimpleTokenizer for capstone purposes\nclass SimpleTokenizer:\n    def encode(self, text: str) -> List[int]:\n        return [ord(c) for c in text if c.isalpha() or c.isspace()] # Simple char-to-int\n    def decode(self, tokens: List[int]) -> str:\n        return \"\".join([chr(t) for t in tokens])\n\nclass MiniLLMEngine:\n    def __init__(self, total_kv_blocks: int = 100, kv_block_size: int = 4, max_batch_size: int = 4):\n        self.tokenizer = SimpleTokenizer()\n        self.request_queue = RequestQueue(self.tokenizer)\n        self.kv_cache_simulator = KVCacheSimulator(total_blocks=total_kv_blocks, block_size=kv_block_size)\n        self.model_runner = MockModelRunner(vocab_size=128) # ASCII range for simple tokenizer\n        self.scheduler = BatchScheduler(self.kv_cache_simulator)\n        self.max_batch_size = max_batch_size\n        self._request_outputs: Dict[str, deque] = {} # request_id -> deque of generated tokens\n\n    def _step(self):\n        # 1. Get pending requests from queue and running requests from scheduler\n        pending_requests = self.request_queue.get_pending_requests()\n        \n        # 2. Schedule a batch for execution\n        current_batch = self.scheduler.schedule(pending_requests, self.max_batch_size)\n\n        if not current_batch:\n            time.sleep(0.01) # Simulate waiting if nothing to do\n            return\n\n        # 3. Execute the model for the current batch\n        generated_tokens = self.model_runner.run_forward_pass(current_batch)\n\n        # 4. Update request states and process generated tokens\n        for req in current_batch:\n            if req.status == \"RUNNING\" and req.request_id in generated_tokens:\n                token_id = generated_tokens[req.request_id]\n                self.scheduler.add_token_to_request(req, token_id)\n                \n                if req.request_id not in self._request_outputs:\n                    self._request_outputs[req.request_id] = deque()\n                self._request_outputs[req.request_id].append(token_id)\n                \n                # Check completion conditions\n                if req.status == \"RUNNING\" and len(req.output_tokens) >= req.max_tokens:\n                    self.scheduler.update_request_status(req, \"COMPLETED\")\n                    self.request_queue.remove_request(req.request_id) # Remove from queue\n            elif req.status == \"ERROR\":\n                # If scheduler marked as ERROR, ensure it's removed from queue\n                self.request_queue.remove_request(req.request_id)\n\n    def generate(self, prompt: str, max_tokens: int = 16) -> Iterator[str]:\n        request = self.request_queue.add_request(prompt, max_tokens)\n        request_id = request.request_id\n        \n        last_output_len = 0\n        while request.status not in (\"COMPLETED\", \"ERROR\"):\n            self._step()\n            if request_id in self._request_outputs:\n                current_output_tokens = list(self._request_outputs[request_id])\n                if len(current_output_tokens) > last_output_len:\n                    new_tokens = current_output_tokens[last_output_len:]\n                    yield self.tokenizer.decode(new_tokens)\n                    last_output_len = len(current_output_tokens)\n            time.sleep(0.005) # Yield control, simulate async wait\n        \n        # Yield any remaining tokens after completion if _step() generated some\n        if request_id in self._request_outputs and len(self._request_outputs[request_id]) > last_output_len:\n            yield self.tokenizer.decode(list(self._request_outputs[request_id])[last_output_len:])\n\n        if request.status == \"ERROR\":\n            raise RuntimeError(f\"Request {request_id} failed with status: {request.status}. Output: {self.tokenizer.decode(list(self._request_outputs.get(request_id, [])))}\")\n",
        "test_behaviors": [
          "Calling `generate` with a prompt should return an iterator that yields generated token chunks over time.",
          "The `generate` method should eventually yield all `max_tokens` for a request unless an error occurs.",
          "Multiple concurrent calls to `generate` should be handled, demonstrating continuous batching via internal `_step` calls."
        ]
      }
    ],
    "integration_test": {
      "description": "Submit multiple concurrent inference requests to the `MiniLLMEngine` and verify that all requests complete, generating the expected number of tokens, and demonstrating throughput benefits of batching.",
      "setup_code": "import threading\n\nengine = MiniLLMEngine(total_kv_blocks=50, kv_block_size=4, max_batch_size=4)\nprompts = [\n    \"Hello, world\",\n    \"How are you\",\n    \"What is vLLM\",\n    \"Tell me a story\",\n    \"Python is great\",\n    \"Artificial intelligence is\",\n    \"Quantum computing will change\",\n    \"The quick brown fox jumps\",\n]\nmax_tokens_per_request = 10\nresults = {}\n\ndef run_inference(prompt, req_id):\n    full_output = []\n    try:\n        for token_chunk in engine.generate(prompt, max_tokens=max_tokens_per_request):\n            full_output.append(token_chunk)\n        results[req_id] = (\"\".join(full_output), \"COMPLETED\", len(\"\".join(full_output)))\n    except RuntimeError as e:\n        results[req_id] = (str(e), \"ERROR\", 0)\n\nthreads = []\nfor i, prompt in enumerate(prompts):\n    thread = threading.Thread(target=run_inference, args=(prompt, f\"req_{i}\"))\n    threads.append(thread)\n    thread.start()\n\nfor thread in threads:\n    thread.join()\n",
      "success_metric": "All submitted requests (`len(prompts)`) should have a status of 'COMPLETED' in the `results` dictionary.",
      "expected_output_check": "For each completed request, the length of its generated output (token count) should be exactly `max_tokens_per_request`."
    }
  }
}
```