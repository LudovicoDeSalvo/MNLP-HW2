from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, TaskType
import os
import torch

def train_model(config, paths, train_dataset, eval_dataset):
    model_name = config["model_name"]
    output_dir = os.path.join(paths['trained_models_dir'], config['output_dir_name'])

    print(f"\n====== Training {model_name} ======")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if torch.cuda.is_available():
        print("CUDA is available. Setting up 8-bit quantization.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print("Warning: CUDA not available. Training in full precision. This will use more memory.")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto"  # Let accelerate handle device placement
    )

    # CORRECTED: Causal LM is the correct task type for decoder-only models.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    def tokenize_supervised(example):
        prompt_ids = tokenizer(example["prompt"], add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(example["target"], add_special_tokens=False)["input_ids"]
        target_ids += [tokenizer.eos_token_id]

        # The combined length is now naturally short due to sentence-splitting
        input_ids = prompt_ids + target_ids
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * len(prompt_ids) + target_ids
        }

    tokenized_train = train_dataset.map(tokenize_supervised, remove_columns=train_dataset.column_names)
    tokenized_eval  = eval_dataset.map(tokenize_supervised, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        learning_rate=2e-4,
        per_device_train_batch_size=1,      # REDUCED to 1
        gradient_accumulation_steps=4,      # ADDED: effective batch size is 1*4=4
        optim="paged_adamw_8bit",           # ADDED: Memory-efficient optimizer
        per_device_eval_batch_size=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training complete. Model and tokenizer saved to {output_dir}")