import re
import os
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score

from src.utils import build_prompt, calculate_cer

EVAL_RESULTS_DIR = "evaluation_results"

# This function is now passed the 'ita' flag from the dataset config
def gemini_judge_score(noisy, predicted, gold, gemini_model, ita=False):
    # ... (rest of the function is identical to the previous version)
    if not gemini_model:
        return -1

    prompt_eng = f"""
    Model correction: {predicted}
    Correct text (gold): {gold}

    Rate from 0 to 5 how accurate the "Model correction" is compared to the "Correct text", based on this scale:
    0: "Completely incorrect or nonsensical."
    1: "Severely flawed. Many major errors."
    2: "Readability or meaning compromised."
    3: "Understandable, but with several minor errors."
    4: "Great, with only small issues (e.g., punctuation)."
    5: "Perfect. Matches the reference text exactly."

    Reply with a single number only (0, 1, 2, 3, 4, or 5).
    """

    prompt_ita = f"""
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

    prompt = prompt_ita if ita else prompt_eng

    try:
        response = gemini_model.generate_content(prompt)
        match = re.search(r'\d+', response.text)
        return int(match.group(0)) if match else -1
    except Exception as e:
        print(f"Error during Gemini scoring: {e}")
        return -1


def evaluate_model(config, dataset_key, eval_dataset, gemini_model, use_gemini_scoring):
    """Evaluates a single fine-tuned model on the evaluation dataset."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = config["model_name"]
    model_path = config["output_dir"]
    prompt_style = config["prompt_style"]
    dataset_config = config["datasets"][dataset_key]
    is_ita = dataset_config.get("ita_language", False)

    print(f"\n====== Evaluating {model_name} from {model_path} on '{dataset_key}' dataset ======")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except OSError:
        print(f"❌ Model not found at {model_path}. Please train the model first or check the path in your config.")
        return None
        
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    for row in tqdm(eval_dataset, desc=f"Evaluating {model_name.split('/')[-1]}"):
        prompt = build_prompt(row["noisy"], prompt_style, is_ita)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=1024)
        
        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        gold = row["target"]
        noisy = row["noisy"]

        gem_score = -1
        if use_gemini_scoring:
            gem_score = gemini_judge_score(noisy, prediction, gold, gemini_model, is_ita)

        results.append({
            "model": model_name,
            "input_text": noisy,
            "predicted_text": prediction,
            "target_text": gold,
            "levenshtein": SequenceMatcher(None, prediction, gold).ratio(),
            "char_error_rate": calculate_cer(prediction, gold),
            "gemini_score": gem_score,
        })

    torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    # Filename now includes the dataset key
    output_filename = os.path.join(EVAL_RESULTS_DIR, f"eval_{model_name.replace('/', '_')}_{dataset_key}.csv")
    results_df.to_csv(output_filename, index=False)
    print(f"✅ Evaluation results saved to {output_filename}")
    
    return results_df


def run_human_vs_gemini_correlation(model_eval_path, human_annotations_path):
    # ... (This function is identical to the previous version, it's now just called with the correct path from main)
    print("\n📊 Checking correlation between Gemini and Human scores...")
    try:
        with open(human_annotations_path, "r", encoding="utf-8") as f:
            human_data = json.load(f)
        eval_df = pd.read_csv(model_eval_path)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}. Cannot perform correlation analysis.")
        return

    eval_df = eval_df[eval_df["gemini_score"] != -1]
    if eval_df.empty:
        print("⚠️ No Gemini-scored samples found in the evaluation file. Cannot perform correlation.")
        return

    human_lookup = { item["ocr"].strip(): item["human_score"] for item in human_data if "human_score" in item }

    human_scores, gemini_scores = [], []
    for _, row in eval_df.iterrows():
        key = row["input_text"].strip()
        if key in human_lookup:
            human_scores.append(human_lookup[key])
            gemini_scores.append(row["gemini_score"])

    if len(human_scores) > 1:
        human_scores_int = [int(s) for s in human_scores]
        gemini_scores_int = [int(s) for s in gemini_scores]
        
        kappa = cohen_kappa_score(human_scores_int, gemini_scores_int)
        correlation = pd.Series(human_scores).corr(pd.Series(gemini_scores))
        
        print(f"✅ Cohen's Kappa between human and Gemini scores: {kappa:.3f}")
        print(f"✅ Pearson Correlation: {correlation:.3f}")
        print(f"(Based on {len(human_scores_int)} overlapping samples)")
    else:
        print(f"⚠️ Only {len(human_scores)} overlapping samples found. Need at least 2 to calculate correlation.")