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
import nltk # <-- Add this import

from src.utils import build_prompt, calculate_cer

EVAL_RESULTS_DIR = "evaluation_results"

# This function is now passed the 'ita' flag from the dataset config
def gemini_judge_score(noisy, predicted, gold, gemini_model, ita=False):
    if not gemini_model:
        return -1

    prompt_eng = f"""
    You are an expert judge of text quality. This is CORRECTED OCR text. Note any mistakes in spelling, grammar, punctuation, or formatting. Check semantinc logic, context consistency and possible hallucinations.

    Here is the data:

    "{predicted}"

    END OF TEXT

    Now provide your rating:
        - 5 (Perfect): The text is excellent, with only trivial errors that do not impact meaning or readibily at all.
        - 4 (Great): The text is readable and mostly correct, but has several minor errors.
        - 3 (Good): The text has some errors that impact readability or meaning but works overall and it's understable.
        - 2 (Poor): The text contains numerous error that make the understanding difficult in some parts.
        - 1 (Failed): The correction is overall wrong or nonsensical.


    Your entire response should be a single number from 1 to 5.
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

def correct_document(doc_text, model, tokenizer, config, dataset_config):
    """Splits a document into sentences, corrects each, and reassembles them."""
    is_ita = dataset_config.get("ita_language", False)
    prompt_style = config["prompt_style"]
    device = model.device

    # Split the document into individual sentences
    sentences = nltk.sent_tokenize(doc_text, language='italian' if is_ita else 'english')
    corrected_sentences = []

    # Process each sentence one-by-one to ensure nothing is missed
    for sent in sentences:
        if not sent.strip():
            continue
            
        prompt = build_prompt(sent, prompt_style, is_ita)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=1024,
                repetition_penalty=1.3,
                no_repeat_ngram_size=4,
                num_beams=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
        prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        # Post-processing to remove leaked special tokens
        prediction = prediction.replace("<|system|>", "").replace("<|user|>", "").strip()

        corrected_sentences.append(prediction)
    
    # Join the corrected sentences with a newline to preserve paragraph structure
    return "\n".join(corrected_sentences)

def evaluate_model(config, dataset_key, eval_docs_df, paths, gemini_model, use_gemini_scoring):
    model_name = config["model_name"]
    model_path = os.path.join(paths['trained_models_dir'], config['output_dir_name'])
    
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = config["model_name"]
    model_path = config["output_dir"]
    dataset_config = config["datasets"][dataset_key]
    is_ita = dataset_config.get("ita_language", False)

    print(f"\n====== Evaluating {model_name} from {model_path} on '{dataset_key}' dataset ======")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except OSError:
        print(f"❌ Model not found at {model_path}. Please train the model first.")
        return None
        
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    for _, row in tqdm(eval_docs_df.iterrows(), total=len(eval_docs_df), desc=f"Evaluating docs for {model_name.split('/')[-1]}"):
        noisy_doc = row["noisy_doc"]
        target_doc = row["target_doc"]

        # Use our new chunking function to get the full prediction
        predicted_doc = correct_document(noisy_doc, model, tokenizer, config, dataset_config)
        
        gem_score = -1
        if use_gemini_scoring:
            gem_score = gemini_judge_score(noisy_doc, predicted_doc, target_doc, gemini_model, is_ita)

        results.append({
            "model": model_name,
            "input_text": noisy_doc,
            "predicted_text": predicted_doc,
            "target_text": target_doc,
            "levenshtein": SequenceMatcher(None, predicted_doc, target_doc).ratio(),
            "char_error_rate": calculate_cer(predicted_doc, target_doc),
            "gemini_score": gem_score,
        })

    torch.cuda.empty_cache()
    
    results_df = pd.DataFrame(results)
    
    eval_dir = paths['evaluation_results_dir']
    os.makedirs(eval_dir, exist_ok=True)
    output_filename = os.path.join(eval_dir, f"eval_{model_name.replace('/', '_')}_{dataset_key}.csv")
    
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