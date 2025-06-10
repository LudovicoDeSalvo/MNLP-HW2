from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import get_peft_model, LoraConfig, TaskType


def train_model(ITA, model_name, output_dir, train_dataset, eval_dataset):
    """Fine-tunes a single decoder-only language model."""
    print(f"\n====== Training {model_name} ======")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    task_type = TaskType.SEQ_2_SEQ_LM if "minerva" in model_name.lower() else TaskType.CAUSAL_LM

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        task_type=task_type,
        lora_dropout=0.05,
        bias="none"
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- TOKENISATION & FILTER  -----------------
    def tokenize_supervised(example):
        if ITA:
            split_key = "Devi pulirlo e correggerlo:"
        else:
            split_key = "You need to clean and correct it:"

        parts = example["text"].split(split_key)
        if len(parts) != 2:
            return {}  # drop malformed sample

        prompt, target = [p.strip() for p in parts]
        prompt += f" {split_key}"

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]

        return {
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(target_ids)),
            "labels": [-100] * len(prompt_ids) + target_ids
        }


    tokenized_train = (
        train_dataset
        .map(tokenize_supervised, remove_columns=train_dataset.column_names)
        .filter(lambda x: len(x.get("input_ids", [])) > 0)
    )
    tokenized_eval  = (
        eval_dataset
        .map(tokenize_supervised, remove_columns=eval_dataset.column_names)
        .filter(lambda x: len(x.get("input_ids", [])) > 0)
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs= 25 ,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
        fp16 = True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
    )

    trainer.train()

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training complete. Model and tokenizer saved to {output_dir}")

