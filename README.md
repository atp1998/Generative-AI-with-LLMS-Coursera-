# Generative-AI-with-LLMS-Coursera-

**Week 1 – Controlled Inference Strategies in Large Language Models**
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


**Week 2 - Fine-tuning LLMs**

Fine-Tuning FLAN-T5 for Dialogue Summarization (Full FT vs PEFT/LoRA)

This repository demonstrates an end-to-end workflow to adapt a pretrained instruction model (FLAN-T5-base) to the DialogSum dialogue summarization task using:

1) **Zero-shot baseline (Original model)**  
2) **Full fine-tuning (Instruct / fully fine-tuned checkpoint)**  
3) **Parameter-Efficient Fine-Tuning (PEFT) with LoRA adapters**

Evaluation is done both:
- Quantitatively using ROUGE (rouge1/rouge2/rougeL/rougeLsum)
- Qualitatively via side-by-side comparison of generated summaries vs human references

Environment Setup

Install the required packages (same versions used in the notebook):

```
# Install dependencies
pip install -U \
  datasets==2.17.0 \
  transformers==4.38.2 \
  accelerate==0.28.0 \
  evaluate==0.4.0 \
  rouge_score==0.1.2 \
  peft==0.3.0

# Evaluation using ROUGE
rouge.compute(
    predictions=predictions,
    references=references,
    use_aggregator=True,
    use_stemmer=True
)
```


**Dataset & Model**

- Dataset: `knkarthick/dialogsum` (DialogSum)
- Backbone model: `google/flan-t5-base` (Seq2Seq)

```
# Zero-Shot Baseline
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt')
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"], 
        max_new_tokens=200,
    )[0], 
    skip_special_tokens=True
)

# Full Fine-tuning
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# PEFT\LoRA Model for fine-tuning
peft_model = get_peft_model(original_model, 
                            lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

```


# Results
| Model                | ROUGE-1    | ROUGE-2    | ROUGE-L    | ROUGE-Lsum |
| -------------------- | ---------- | ---------- | ---------- | ---------- |
| Original (Zero-shot) | 0.2334     | 0.0760     | 0.2015     | 0.2015     |
| Full Fine-Tuning     | **0.4216** | **0.1804** | **0.3384** | **0.3384** |
| PEFT (LoRA)          | 0.4081     | 0.1633     | 0.3251     | 0.3249     |

| Model   | R-1 Gain  | R-2 Gain  | R-L Gain  |
| ------- | --------- | --------- | --------- |
| Full FT | +18.82 pp | +10.43 pp | +13.70 pp |
| PEFT    | +17.47 pp | +8.73 pp  | +12.36 pp |

