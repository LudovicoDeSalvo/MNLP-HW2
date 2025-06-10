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
    DataCollatorForSeq2Seq,
    set_seed,
    EarlyStoppingCallback
)
from huggingface_hub import login
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from wtpsplit import SaT as SaTSplitter
from sentence_transformers import SentenceTransformer, util
from peft import get_peft_model, LoraConfig, TaskType

from src.utils import set_all_seeds, login_to_huggingface, configure_gemini
from src.data_prep import prepare_dataset
from src.train import train_model
from src.eval import evaluate_models, run_pairwise_comparison


# --- 1. SETUP AND CONFIGURATION ---

#FLAGS
MAX_TOKENS = 256
SENTENCE_SPLITTING = False
ITA = False
INFERENCE_ONLY = False
MINERVA_FIRST = False

# Define model checkpoints to be trained and evaluated. Models for inference takes the models
#   directly from Hugging Faces
if MINERVA_FIRST:
    MODELS_TO_TRAIN = {
        "sapienzanlp/Minerva-350M-base-v1.0": "./ocr_model_minerva350m",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "./ocr_model_tinyllama",
    }
    MODELS_FOR_INFERENCE = {
        "sapienzanlp/Minerva-350M-base-v1.0": "sapienzanlp/Minerva-350M-base-v1.0",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    }
else:
    MODELS_TO_TRAIN = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "./ocr_model_tinyllama",
        "sapienzanlp/Minerva-350M-base-v1.0": "./ocr_model_minerva350m",
    }
    MODELS_FOR_INFERENCE = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "sapienzanlp/Minerva-350M-base-v1.0": "sapienzanlp/Minerva-350M-base-v1.0",        
    }


# Define patterns for synthetic OCR noise generation
OCR_NOISE_PATTERNS = [
    (r'm', 'rn'), (r'\bi\b', '1'), (r'\bo\b', '0'), (r'\bé\b', 'e'),
    (r'\bè\b', 'e'), (r'\bì\b', 'i'), (r'\bù\b', 'u'), (r'rn', 'm'), (r'\b1\b', 'i')
]


# Set a CUDA environment variable for easier debugging if needed
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

def main(inference_only=True):

    set_all_seeds(42)
    login_to_huggingface()
    gemini_model = configure_gemini()

    if ITA:
        original_path="dataset/ita/original_ocr.json"
        cleaned_path="dataset/ita/cleaned.json"
    else:
        original_path="dataset/eng/the_vampyre_ocr.json"
        cleaned_path="dataset/eng/the_vampyre_clean.json"

    all_results = []
    model_dict = MODELS_TO_TRAIN if not inference_only else MODELS_FOR_INFERENCE

    for model_name, model_path in model_dict.items():

        train_ds, eval_ds = prepare_dataset(
            SENTENCE_SPLITTING,
            ITA,
            original_path,
            cleaned_path,
            model_name
        )

        if not INFERENCE_ONLY:
            train_model(ITA, model_name, model_path, train_ds, eval_ds)

        # keep the eval datasets so we can merge them later if you want
        result_df = evaluate_models(ITA, MAX_TOKENS, {model_name: model_path}, eval_ds, gemini_model)
        all_results.append(result_df)

    
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv("ocr_eval_results_causal.csv", index=False)
    print("\n✅ Evaluation results saved to ocr_eval_results_causal.csv")

    pairwise_df = run_pairwise_comparison(ITA, results_df, gemini_model, model_dict)
    if not pairwise_df.empty:
        pairwise_df.to_csv("ocr_pairwise_comparison.csv", index=False)
        print("✅ Pairwise comparison saved to ocr_pairwise_comparison.csv")

    summarize_results(results_df, pairwise_df)
    print("\n🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    main(inference_only=INFERENCE_ONLY)