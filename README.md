# Generative-AI-with-LLMS-Coursera-

Week 1 – Controlled Inference Strategies in Large Language Models
Zero-Shot, One-Shot, and Few-Shot Prompting for Dialogue Summarization

In Week 1, I conducted a structured study on inference-time conditioning strategies in pretrained transformer-based large language models (LLMs), focusing on zero-shot, one-shot, and few-shot prompting paradigms for dialogue summarization.

The experiments were implemented using HuggingFace Transformers with PyTorch backend, operating purely in inference mode (no parameter updates).

🔹 Zero-Shot Inference
Summarize the following dialogue:
[Dialogue]

🔹 One-Shot Inference
Example:
Dialogue: ...
Summary: ...

Now summarize:
[New Dialogue]

🔹 Few-Shot Inference
Multiple demonstration pairs included.

Tokenizetion : 
inputs = tokenizer(dialogue, return_tensors="pt")

model.generate(
    inputs["input_ids"],
    max_new_tokens=50
)

Findings : Zero-shot acts as a baseline for evaluating prompt sensitivity. One shot and few-shot prompting effectively biases the internal representation toward task-specific distribution without weight updates.
