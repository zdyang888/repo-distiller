# Understanding vLLM

The system processes incoming requests through an `LLMEngine`, which uses an `InputProcessor` to prepare them and an `OutputProcessor` to format results. The core of the system is the `EngineCore`, which coordinates a `Scheduler` and an `Executor`. The `Scheduler` manages the lifecycle of requests and their KV cache blocks (via PagedAttention), while the `Executor` is responsible for running the model on the underlying hardware, performing the actual forward passes.

## Contents

| Notebook | Description |
|----------|-------------|
| [01. Request Input & Output Processing](notebooks/01_request_input_output_processing.ipynb) | Students will build simplified versions of an `InputProcessor` to tokenize prompts and an `OutputProcessor` to detokenize generated tokens and manage streaming output. |
| [02. PagedAttention: Efficient KV Cache Management](notebooks/02_pagedattention_efficient_kv_cache_manage.ipynb) | Students will simulate the core mechanics of PagedAttention, managing fixed-size KV cache blocks for multiple sequences, and understanding how it reduces memory fragmentation to improve throughput. |
| [03. The LLM Model Executor](notebooks/03_the_llm_model_executor.ipynb) | Students will build a mock `Executor` that simulates running a model's forward pass, accepting batches of token IDs and returning mock logits, focusing on the interface between the scheduler and the model. |
| [04. Request Scheduling and Continuous Batching](notebooks/04_request_scheduling_and_continuous_batchi.ipynb) | Students will implement a simplified `Scheduler` that manages multiple incoming requests, prioritizes them, and constructs batches for the `Executor` using continuous batching logic, interacting with the KV cache manager. |
| [05. Orchestrating the Inference Engine Core](notebooks/05_orchestrating_the_inference_engine_core.ipynb) | Students will integrate the `Scheduler`, `Executor`, and `Input/Output Processors` into a basic `EngineCore`, demonstrating the central coordination of inference requests from intake to output. |
| [06. The User-Facing LLM Engine API](notebooks/06_the_user_facing_llm_engine_api.ipynb) | Students will create a high-level `LLMEngine` that provides a user-friendly, streaming interface for submitting prompts and receiving responses, utilizing the `EngineCore`. |

## Capstone Project

**mini-vLLM**: Students will build a simplified, end-to-end LLM inference server that leverages the key concepts of vLLM: request processing, efficient KV cache management, continuous batching, and model execution. This capstone will solidify their understanding of how these components integrate to achieve high-throughput LLM serving.

See `capstone/` for instructions, starter code, and tests.

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook
```