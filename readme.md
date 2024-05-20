# Evaluation Criteria Generator

This project aims to generate an evaluation dataset that can evaluate a RAG solution in a financial context.

## 1. adhoc_chat

Generates categories of instructions that can be used in main to actually generate the test data.
By creating categories dynamically, we can find new and challenging evaluation data.

[adhoc_chat](src/adhoc_chat.py)

## 2. main.py

Actually generates the evaluation data. Can be seeded with results from adhoc_chat.

See 'evaluation_category_prompts' for the prompts

[main](src/main.py)
