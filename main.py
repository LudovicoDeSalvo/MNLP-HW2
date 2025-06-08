# === FULL UPDATED SCRIPT: OCR Training with Minerva + TinyLLaMA (Decoder-Only) ===

import os
import random
import numpy as np
import torch
import json
import re
import pandas as pd
from difflib import SequenceMatcher
from sklearn.model_selection import train_test_split
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed
)
from huggingface_hub import login
import google.generativeai as genai

# Login to HuggingFace Hub (use environment variable for safety)
try:
    with open("general_utils/hf_token.txt") as f:
        login(token=f.read().strip())
except Exception as e:
    print(f"⚠️ HuggingFace login failed: {e}")


# Set seed for reproducibility
def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_all_seeds(42)

# Define OCR noise injection patterns
OCR_NOISE_PATTERNS = [
    (r'm', 'rn'), (r'\bi\b', '1'), (r'\bo\b', '0'), (r'\bé\b', 'e'),
    (r'\bè\b', 'e'), (r'\bì\b', 'i'), (r'\bù\b', 'u'), (r'rn', 'm'), (r'\b1\b', 'i')
]

def inject_ocr_noise(text, num_errors=2):
    noisy = text
    for _ in range(num_errors):
        pattern, replacement = random.choice(OCR_NOISE_PATTERNS)
        noisy = re.sub(pattern, replacement, noisy, count=1)
    return noisy

# Load and augment dataset
with open("dataset/ita/original_ocr.json", "r", encoding="utf-8") as f:
    original = json.load(f)
with open("dataset/ita/cleaned.json", "r", encoding="utf-8") as f:
    cleaned = json.load(f)

examples = []
for k, clean in cleaned.items():
    clean = clean.strip()
    noisy = original.get(k, "").strip()
    if clean and noisy:
        prompt = f"OCR: {noisy}\nClean:"
        full_text = f"{prompt} {clean}"
        examples.append({"text": full_text, "source": "real"})
        for _ in range(2):
            synthetic = inject_ocr_noise(clean)
            prompt = f"OCR: {synthetic}\nClean:"
            full_text = f"{prompt} {clean}"
            examples.append({"text": full_text, "source": "synthetic"})

random.shuffle(examples)
train_data, eval_data = train_test_split(examples, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))

# Define model checkpoints
MODELS = {
    "sapienzanlp/Minerva-350M-base-v1.0": "./ocr_model_minerva350m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "./ocr_model_tinyllama"
}

# Train each model
for model_name, output_dir in MODELS.items():
    print(f"\n====== Training {model_name} ======")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        greater_is_better=False,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()

# Evaluation
with open("general_utils/google_api.txt") as f:
    genai.configure(api_key=f.read().strip())
gemini_model = genai.GenerativeModel("gemini-2.5-flash-preview-05-20")

def levenshtein_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def gemini_judge_score(noisy, predicted, gold):
    prompt = f"""
    Testo OCR (noisy): {noisy}
    Correzione del modello: {predicted}
    Testo corretto: {gold}

    Valuta da 0 a 5 quanto è accurata la correzione rispetto al testo corretto, basandoti su questa metrica:
    0 : "Completamente errato o privo di senso. Illeggibile."
    1 : "Gravemente compromesso. Molti errori gravi, difficilmente comprensibile."
    2 : "Leggibilità o significato compromessi."
    3 : "Comprensibile, ma con diversi errori minori o qualche errore grave."
    4 : "Ottimo, con solo piccoli difetti (es. punteggiatura, maiuscole/minuscole o rari refusi)."
    5 : "Perfetto. Completamente corretto e corrispondente esattamente al testo di riferimento."
    Rispondi solo con un numero.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return int(response.text.strip().split("\n")[0])
    except Exception as e:
        print(f"Errore Gemini: {e}")
        return -1

results = []
for model_name, path in MODELS.items():
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to("cuda" if torch.cuda.is_available() else "cpu")

    for row in tqdm(eval_dataset):
        full_text = row["text"]
        noisy = full_text.split("\nClean:")[0].replace("OCR: ", "").strip()
        gold = full_text.split("\nClean:")[1].strip()

        input_ids = tokenizer(full_text.split("\nClean:")[0] + "\nClean:", return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=512)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("\nClean:")[-1].strip()

        lev = levenshtein_ratio(prediction, gold)
        gem_score = gemini_judge_score(noisy, prediction, gold)

        results.append({
            "model": model_name,
            "input_text": noisy,
            "predicted_text": prediction,
            "target_text": gold,
            "levenshtein": lev,
            "gemini_score": gem_score,
            "source": row.get("source", "unknown")
        })

pd.DataFrame(results).to_csv("ocr_eval_results_causal.csv", index=False)

# === Pairwise Gemini Comparison Between Models ===
print("🔍 Starting pairwise Gemini comparison between models")
from collections import defaultdict

# Load both models' predictions per input
model_preds = defaultdict(dict)

for row in results:
    key = (row["input_text"], row["target_text"])
    model_preds[key][row["model"]] = row["predicted_text"]

comparison_results = []

for (noisy, gold), preds in tqdm(model_preds.items()):
    if len(preds) < 2:
        continue  # need both models to compare
    minerva_pred = preds.get("sapienzanlp/Minerva-350M-base-v1.0", "")
    tinyllama_pred = preds.get("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "")

    compare_prompt = f"""
    Testo OCR (noisy): {noisy}

    Predizione A (Minerva): {minerva_pred}
    Predizione B (TinyLLaMA): {tinyllama_pred}
    Testo corretto: {gold}

    Quale predizione è più vicina al testo corretto?
    Rispondi solo con una delle seguenti opzioni: A / B / TIE
    """
    try:
        judge_response = gemini_model.generate_content(compare_prompt).text.strip().split("")[0].upper()
    except Exception as e:
        print(f"Errore Gemini pairwise: {e}")
        judge_response = "ERROR"

    comparison_results.append({
        "input_text": noisy,
        "target_text": gold,
        "minerva_pred": minerva_pred,
        "tinyllama_pred": tinyllama_pred,
        "gemini_judgment": judge_response
    })

pd.DataFrame(comparison_results).to_csv("ocr_pairwise_comparison.csv", index=False)
print("✅ Pairwise comparison saved to ocr_pairwise_comparison.csv")

# === Best Model Summary ===
print("🏆 Selecting best model based on metrics")

# Load results
results_df = pd.read_csv("ocr_eval_results_causal.csv")
pairwise_df = pd.read_csv("ocr_pairwise_comparison.csv")

# 1. Levenshtein + Gemini mean scores
mean_scores = results_df.groupby("model")[["levenshtein", "gemini_score"]].mean()
print("📈 Average Scores Per Model:")
print(mean_scores)

# 2. Count pairwise wins
win_counts = {"minerva": 0, "tinyllama": 0, "tie": 0}
for val in pairwise_df["gemini_judgment"].str.upper():
    if val == "A":
        win_counts["minerva"] += 1
    elif val == "B":
        win_counts["tinyllama"] += 1
    elif val == "TIE":
        win_counts["tie"] += 1

print("🤖 Pairwise Gemini Win Counts:")
print(win_counts)

# 3. Normalized decision rule
from sklearn.preprocessing import MinMaxScaler

# Prepare raw values
gemini_raw = [mean_scores.loc["sapienzanlp/Minerva-350M-base-v1.0", "gemini_score"],
              mean_scores.loc["TinyLlama/TinyLlama-1.1B-Chat-v1.0", "gemini_score"]]
wins_raw = [win_counts["minerva"], win_counts["tinyllama"]]

scaler = MinMaxScaler()
gemini_scaled = scaler.fit_transform(np.array(gemini_raw).reshape(-1, 1)).flatten()
wins_scaled = scaler.fit_transform(np.array(wins_raw).reshape(-1, 1)).flatten()

score = {
    "minerva": 0.7 * gemini_scaled[0] + 0.3 * wins_scaled[0],
    "tinyllama": 0.7 * gemini_scaled[1] + 0.3 * wins_scaled[1]
}
best = max(score, key=score.get)
print(f"✅ Best model overall: {best.upper()} based on Gemini score + pairwise wins")
print("✅ Evaluation complete. Results saved to ocr_eval_results_causal.csv")

# === Human-Gemini Correlation Analysis ===
print("📊 Loading human annotations for correlation check")
from scipy.stats import cohen_kappa_score

with open("general_utils/human_annotations.json", "r", encoding="utf-8") as f:
    human_data = json.load(f)

human_scores = []
gemini_scores = []

# match key = (input_text, target_text)
human_lookup = {
    (item["input_text"].strip(), item["target_text"].strip()): item["score"]
    for item in human_data if item["score"] is not None
}

for row in results:
    key = (row["input_text"].strip(), row["target_text"].strip())
    if key in human_lookup:
        human_scores.append(human_lookup[key])
        gemini_scores.append(row["gemini_score"])

if human_scores:
    kappa = cohen_kappa_score(human_scores, gemini_scores)
    print(f"✅ Cohen's kappa between human and Gemini scores: {kappa:.3f}")
else:
    print("⚠️ No overlapping samples between Gemini and human annotations.")