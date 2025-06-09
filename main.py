# === FULL UPDATED & FIXED SCRIPT: OCR Training with Minerva + TinyLLaMA (Decoder-Only) ===

import os
import random
import re
import json
import numpy as np
import pandas as pd
import torch
import google.generativeai as genai
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
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from transformers import EarlyStoppingCallback


# --- 1. SETUP AND CONFIGURATION ---

# Set a CUDA environment variable for easier debugging if needed
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Define model checkpoints to be trained and evaluated
MODELS_TO_TRAIN = {
    "sapienzanlp/Minerva-350M-base-v1.0": "./ocr_model_minerva350m",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "./ocr_model_tinyllama"
}

# Define patterns for synthetic OCR noise generation
OCR_NOISE_PATTERNS = [
    (r'm', 'rn'), (r'\bi\b', '1'), (r'\bo\b', '0'), (r'\bé\b', 'e'),
    (r'\bè\b', 'e'), (r'\bì\b', 'i'), (r'\bù\b', 'u'), (r'rn', 'm'), (r'\b1\b', 'i')
]

def set_all_seeds(seed=42):
    """Sets seeds for all relevant libraries to ensure reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    # The following two lines are important for deterministic behavior on GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def login_to_huggingface(token_path="general_utils/hf_token.txt"):
    """Logs into HuggingFace Hub using a token from a file."""
    try:
        with open(token_path) as f:
            token = f.read().strip()
            login(token=token)
            print("✅ Successfully logged into HuggingFace Hub.")
    except Exception as e:
        print(f"⚠️ HuggingFace login failed: {e}. You may not be able to download models.")

def configure_gemini(api_key_path="general_utils/google_api.txt"):
    """Configures the Gemini API and returns the generative model."""
    try:
        with open(api_key_path) as f:
            api_key = f.read().strip()
            genai.configure(api_key=api_key)
            print("✅ Successfully configured Gemini API.")
            return genai.GenerativeModel("gemini-2.5-flash-preview-05-20")
    except Exception as e:
        print(f"⚠️ Failed to configure Gemini API: {e}. Evaluation with Gemini will fail.")
        return None

# --- 2. DATA PREPARATION ---

def inject_ocr_noise(text, num_errors=2):
    """Applies a number of random OCR-like errors to a string."""
    noisy = text
    for _ in range(num_errors):
        pattern, replacement = random.choice(OCR_NOISE_PATTERNS)
        noisy = re.sub(pattern, replacement, noisy, count=1)
    return noisy

def prepare_dataset(original_path, cleaned_path, test_size=0.2, random_state=42):
    """Loads, augments, and splits the dataset for training and evaluation."""
    with open(original_path, "r", encoding="utf-8") as f:
        original = json.load(f)
    with open(cleaned_path, "r", encoding="utf-8") as f:
        cleaned = json.load(f)

    examples = []
    for k, clean_text in cleaned.items():
        clean_text = clean_text.strip()
        noisy_text = original.get(k, "").strip()

        # Add real examples
        if clean_text and noisy_text:
            prompt = f"Questo è testo OCR: {noisy_text}\nDevi pulirlo e correggerlo:"
            full_text = f"{prompt} {clean_text}"
            examples.append({"text": full_text, "source": "real"})

            # Add synthetically augmented examples
            for _ in range(2):
                synthetic_noisy = inject_ocr_noise(clean_text)
                prompt = f"Questo è testo OCR: {synthetic_noisy}\nDevi pulirlo e correggerlo:"
                full_text = f"{prompt} {clean_text}"
                examples.append({"text": full_text, "source": "synthetic"})

    random.shuffle(examples)
    train_data, eval_data = train_test_split(examples, test_size=test_size, random_state=random_state)
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    print(f"📚 Dataset prepared: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset

# --- 3. MODEL TRAINING ---

def train_model(model_name, output_dir, train_dataset, eval_dataset):
    """Fine-tunes a single language model."""
    print(f"\n====== Training {model_name} ======")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    def tokenize_function(batch):
        return tokenizer(batch["text"], padding="longest", truncation=True, max_length=512)

    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=25,
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

    # ✅ Save model and tokenizer explicitly
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"✅ Training complete. Model and tokenizer saved to {output_dir}")


# --- 4. EVALUATION ---

def gemini_judge_score(noisy, predicted, gold, gemini_model):
    """Uses Gemini to score the quality of a correction on a 0-5 scale."""
    if not gemini_model: return -1
    
    prompt = f"""
    Correzione del modello: {predicted}
    Testo corretto (gold): {gold}

    Valuta da 0 a 5 quanto è accurata la "Correzione del modello" rispetto al "Testo corretto", basandoti su questa metrica:
    0: "Completamente errato o privo di senso."
    1: "Gravemente compromesso. Molti errori gravi."
    2: "Leggibilità o significato compromessi."
    3: "Comprensibile, ma con diversi errori minori."
    4: "Ottimo, con solo piccoli difetti (es. punteggiatura)."
    5: "Perfetto. Corrisponde esattamente al testo di riferimento."
    
    Rispondi solo con un singolo numero (0, 1, 2, 3, 4, o 5).
    """
    try:
        response = gemini_model.generate_content(prompt)
        # FIXED: Use regex for robust parsing of the numeric score
        match = re.search(r'\d+', response.text)
        return int(match.group(0)) if match else -1
    except Exception as e:
        print(f"Error during Gemini scoring: {e}")
        return -1

def evaluate_models(models_dict, eval_dataset, gemini_model):
    """Evaluates the fine-tuned models on the evaluation dataset."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for model_name, model_path in models_dict.items():
        print(f"\n====== Evaluating {model_name} from {model_path} ======")
        
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for row in tqdm(eval_dataset, desc=f"Evaluating {model_name.split('/')[-1]}"):
            full_text = row["text"]

            match = re.search(r"testo OCR:\s*(.*?)\s*Devi pulirlo e correggerlo:\s*(.*)", full_text, re.DOTALL | re.IGNORECASE)

            if match:
                noisy = match.group(1).strip()
                gold = match.group(2).strip()
            else:
                print("⚠️ Format error in text:", full_text[:100])
                continue

            prompt = f"Questo è un testo OCR: {noisy}\nDevi pulirlo e correggerlo:"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)

            prediction = tokenizer.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

            lev_ratio = SequenceMatcher(None, prediction, gold).ratio()
            gem_score = gemini_judge_score(noisy, prediction, gold, gemini_model)

            results.append({
                "model": model_name,
                "input_text": noisy,
                "predicted_text": prediction,
                "target_text": gold,
                "levenshtein": lev_ratio,
                "gemini_score": gem_score,
                "source": row.get("source", "unknown")
            })
        torch.cuda.empty_cache()

    return pd.DataFrame(results)


# --- 5. PAIRWISE COMPARISON ---

def run_pairwise_comparison(results_df, gemini_model):
    """Uses Gemini to perform a pairwise A/B test between the two models."""
    if not gemini_model or len(results_df['model'].unique()) < 2:
        print("⚠️ Skipping pairwise comparison (requires Gemini and >1 model).")
        return pd.DataFrame()

    print("\n🔍 Starting pairwise Gemini comparison between models")
    model_preds = defaultdict(dict)
    for _, row in results_df.iterrows():
        key = (row["input_text"], row["target_text"])
        model_preds[key][row["model"]] = row["predicted_text"]

    comparison_results = []
    model_A_name, model_B_name = list(MODELS_TO_TRAIN.keys())

    for (noisy, gold), preds in tqdm(model_preds.items(), desc="Pairwise Judging"):
        if len(preds) < 2:
            continue

        minerva_pred = preds.get(model_A_name, "")
        tinyllama_pred = preds.get(model_B_name, "")

        compare_prompt = f"""

        Predizione A (Minerva): {minerva_pred}
        Predizione B (TinyLLaMA): {tinyllama_pred}
        Testo corretto (gold): {gold}

        Quale predizione è più vicina al testo corretto?
        Rispondi solo con una delle seguenti opzioni: A / B / TIE
        """
        try:
            response = gemini_model.generate_content(compare_prompt)
            # FIXED: Correctly parse the single-character response
            response_text = response.text.strip().upper()
            judge_response = response_text[0] if response_text in ["A", "B", "TIE"] else "ERROR"
        except Exception as e:
            print(f"Error during Gemini pairwise judging: {e}")
            judge_response = "ERROR"

        comparison_results.append({
            "input_text": noisy, "target_text": gold,
            "minerva_pred": minerva_pred, "tinyllama_pred": tinyllama_pred,
            "gemini_judgment": judge_response
        })

    return pd.DataFrame(comparison_results)

# --- 6. ANALYSIS AND SUMMARY ---

def summarize_results(results_df, pairwise_df):
    """Analyzes and prints a summary of the best performing model."""
    print("\n🏆 Selecting best model based on metrics")

    # 1. Levenshtein + Gemini mean scores
    mean_scores = results_df.groupby("model")[["levenshtein", "gemini_score"]].mean()
    print("\n📈 Average Scores Per Model:")
    print(mean_scores)
    
    model_A_name, model_B_name = list(MODELS_TO_TRAIN.keys())

    if not pairwise_df.empty:
        # 2. Count pairwise wins
        win_counts = pairwise_df["gemini_judgment"].value_counts().to_dict()
        minerva_wins = win_counts.get("A", 0)
        tinyllama_wins = win_counts.get("B", 0)
        
        print("\n🤖 Pairwise Gemini Win Counts:")
        print(f" - Minerva Wins: {minerva_wins}")
        print(f" - TinyLLaMA Wins: {tinyllama_wins}")
        print(f" - Ties: {win_counts.get('TIE', 0)}")

        # 3. Normalized decision rule
        scaler = MinMaxScaler()
        gemini_raw = np.array([
            mean_scores.loc[model_A_name, "gemini_score"],
            mean_scores.loc[model_B_name, "gemini_score"]
        ]).reshape(-1, 1)
        
        wins_raw = np.array([minerva_wins, tinyllama_wins]).reshape(-1, 1)

        gemini_scaled = scaler.fit_transform(gemini_raw).flatten()
        wins_scaled = scaler.fit_transform(wins_raw).flatten()

        score = {
            "minerva": 0.7 * gemini_scaled[0] + 0.3 * wins_scaled[0],
            "tinyllama": 0.7 * gemini_scaled[1] + 0.3 * wins_scaled[1]
        }
        best_model_name = max(score, key=score.get)
        print(f"\n✅ Best model overall: {best_model_name.upper()} based on a weighted score of Gemini ratings and pairwise wins.")
    else:
        best_model_name = mean_scores['gemini_score'].idxmax()
        print(f"\n✅ Best model based on mean Gemini score: {best_model_name}")

def analyze_human_correlation(results_df, annotations_path="general_utils/human_annotations.json"):
    print("\n📊 Loading human annotations for correlation check")
    try:
        with open(annotations_path, "r", encoding="utf-8") as f:
            human_data = json.load(f)
    except FileNotFoundError:
        print("⚠️ Human annotations file not found. Skipping correlation analysis.")
        return

    # Auto-detect key names
    sample_keys = set(human_data[0].keys())
    if "input_text" in sample_keys and "target_text" in sample_keys:
        key_fn = lambda x: (x["input_text"].strip(), x["target_text"].strip())
    elif "ocr" in sample_keys and "cleaned" in sample_keys:
        key_fn = lambda x: (x["ocr"].strip(), x["cleaned"].strip())
    else:
        print("❌ Unrecognized key structure in human annotations.")
        return

    human_lookup = {
        key_fn(item): item["human_score"]
        for item in human_data
        if item.get("human_score") is not None
    }

    human_scores, gemini_scores = [], []
    for _, row in results_df.iterrows():
        key = (row["input_text"].strip(), row["target_text"].strip())
        if key in human_lookup:
            human_scores.append(human_lookup[key])
            gemini_scores.append(row["gemini_score"])

    if human_scores and len(human_scores) > 1:
        kappa = cohen_kappa_score(human_scores, gemini_scores)
        print(f"✅ Cohen's Kappa between human and Gemini scores: {kappa:.3f} (based on {len(human_scores)} overlapping samples)")
    else:
        print("⚠️ No overlapping samples found between Gemini and human annotations to calculate Kappa.")


# --- 7. MAIN EXECUTION ---

def main(inference_only=True):  # Add this argument
    set_all_seeds(42)
    login_to_huggingface()
    gemini_model = configure_gemini()

    train_ds, eval_ds = prepare_dataset(
        original_path="dataset/ita/original_ocr.json",
        cleaned_path="dataset/ita/cleaned.json"
    )

    if not inference_only:
        for model_name, output_dir in MODELS_TO_TRAIN.items():
            train_model(model_name, output_dir, train_ds, eval_ds)

    results_df = evaluate_models(MODELS_TO_TRAIN, eval_ds, gemini_model)
    results_df.to_csv("ocr_eval_results_causal.csv", index=False)
    print("\n✅ Evaluation results saved to ocr_eval_results_causal.csv")

    pairwise_df = run_pairwise_comparison(results_df, gemini_model)
    if not pairwise_df.empty:
        pairwise_df.to_csv("ocr_pairwise_comparison.csv", index=False)
        print("✅ Pairwise comparison saved to ocr_pairwise_comparison.csv")

    summarize_results(results_df, pairwise_df)
    analyze_human_correlation(results_df)

    print("\n🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    main(inference_only=False)