# Understanding vLLM: Fast LLM Inference and Serving

The vLLM system receives user requests through its LLMEngine, which then dispatches them to an EngineCore via an EngineCoreClient. The EngineCore orchestrates the entire inference process, utilizing a Scheduler to manage incoming requests, prioritize them, and allocate KV cache blocks using PagedAttention. An Executor component then takes the scheduled batches and runs the actual model inference, potentially across multiple devices, while interacting with the KV cache and returning outputs to the Scheduler, which are then processed back to the user.

## Contents

| Notebook | Description |
|----------|-------------|
| [01. Interacting with vLLM: The User API](notebooks/01_interacting_with_vllm_the_user_api.ipynb) | Learn how users interact with vLLM by sending requests and receiving responses, understanding the LLMEngine's role as the primary interface to the system. |
| [02. The Heart of vLLM: Request Orchestration with EngineCore](notebooks/02_the_heart_of_vllm_request_orchestration_.ipynb) | Understand how the EngineCore orchestrates the entire inference lifecycle, dispatching requests and coordinating between scheduling and execution components. |
| [03. Efficient Request Handling: The Scheduler](notebooks/03_efficient_request_handling_the_scheduler.ipynb) | Dive into how the Scheduler component manages the queue of incoming requests, prioritizes them, and prepares batches for model execution, considering resource constraints. |
| [04. Model Execution: The Executor](notebooks/04_model_execution_the_executor.ipynb) | Explore how the Executor component is responsible for loading the LLM model and performing the actual forward pass on scheduled batches of requests. |
| [05. Memory Efficiency: PagedAttention KV Cache](notebooks/05_memory_efficiency_pagedattention_kv_cach.ipynb) | Understand the advanced PagedAttention technique for managing KV cache memory, enabling higher throughput and efficient resource utilization by treating KV cache as a paged memory system. |

## Capstone Project

**mini-vLLM Inference Server**: Students will build a simplified, end-to-end LLM inference server that processes user requests, schedules them based on availability, executes a mock model for token generation, and manages a basic KV cache using PagedAttention principles, synthesizing all core vLLM concepts.

See `capstone/` for instructions, starter code, and tests.

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook
```