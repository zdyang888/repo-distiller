# Understanding nanoGPT: Minimalist GPT Training Framework

nanoGPT implements a clear and efficient version of the GPT-2 transformer architecture, focusing on making large language model training accessible. Students will learn to process text data, build the core GPT model, orchestrate its training loop, and finally generate text from a trained model.

## Contents

| Notebook | Description |
|----------|-------------|
| [01. Preparing Text Data for LLMs](notebooks/01_preparing_text_data_for_llms.ipynb) | Students will learn how to tokenize raw text and convert it into a binary format suitable for efficient loading by an LLM, mimicking the `nanoGPT` data preparation pipeline. |
| [02. Deconstructing the GPT Model Architecture](notebooks/02_deconstructing_the_gpt_model_architectur.ipynb) | Students will build a simplified version of the core components of the GPT model, including attention, MLP, and the overall block structure, understanding how they fit together. |
| [03. Orchestrating the LLM Training Loop](notebooks/03_orchestrating_the_llm_training_loop.ipynb) | Students will implement a basic training loop for a language model, covering data loading, forward and backward passes, optimizer steps, and basic logging. |
| [04. Generating Text with a Trained GPT](notebooks/04_generating_text_with_a_trained_gpt.ipynb) | Students will learn how to load a trained GPT model and use it to autoregressively generate new text sequences given a starting prompt. |

## Capstone Project

**mini-nanoGPT: End-to-End LLM Pipeline**: Students will build a simplified, end-to-end version of `nanoGPT`, capable of processing a small dataset, defining a minimalist GPT model, training it, and generating text. This project synthesizes all concepts learned throughout the course.

See `capstone/` for instructions, starter code, and tests.

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook
```