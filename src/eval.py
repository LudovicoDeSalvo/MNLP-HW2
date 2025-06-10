import re
import torch
import google.generativeai as genai
from difflib import SequenceMatcher
from tqdm import tqdm
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import defaultdict

from src.utils import build_prompt

def gemini_judge_score(ITA, noisy, predicted, gold, gemini_model):

    if not gemini_model: return -1
    
    if ITA:
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
    else:
        prompt = f"""
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

    try:
        response = gemini_model.generate_content(prompt)
        match = re.search(r'\d+', response.text)
        return int(match.group(0)) if match else -1
    except Exception as e:
        print(f"Error during Gemini scoring: {e}")
        return -1

def evaluate_models(ITA, MAX_TOKENS, model_dict, eval_dataset, gemini_model):
    """Evaluates the fine-tuned models on the evaluation dataset."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_name, model_path in model_dict.items():
        print(f"\n====== Evaluating {model_name} from {model_path} ======")

        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        generation_args = {
            "max_new_tokens": MAX_TOKENS,
            "do_sample": False,
            "num_beams": 4,
            "pad_token_id": tokenizer.eos_token_id
        }

        for row in tqdm(eval_dataset, desc=f"Evaluating {model_name.split('/')[-1]}"):
            prompt = row["prompt"]
            gold = row["target"]
            noisy = row["noisy"]

            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_args
                )

            new_tokens = output_ids[0][inputs['input_ids'].shape[1]:]
            prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            lev_ratio = SequenceMatcher(None, prediction, gold).ratio()
            gem_score = gemini_judge_score(ITA, noisy, prediction, gold, gemini_model)

            results.append({
                "model": model_name,
                "input_text": noisy,
                "predicted_text": prediction,
                "target_text": gold,
                "levenshtein": lev_ratio,
                "gemini_score": gem_score,
            })

        torch.cuda.empty_cache()

    return pd.DataFrame(results)



# --- 5. PAIRWISE COMPARISON ---

def run_pairwise_comparison(ITA, results_df, gemini_model, model_dict):
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
    model_A_name, model_B_name = list(model_dict.keys())


    for (noisy, gold), preds in tqdm(model_preds.items(), desc="Pairwise Judging"):
        if len(preds) < 2:
            continue

        minerva_pred = preds.get(model_A_name, "")
        tinyllama_pred = preds.get(model_B_name, "")

        if ITA:
            compare_prompt = f"""

            Predizione A (Minerva): {minerva_pred}
            Predizione B (TinyLLaMA): {tinyllama_pred}
            Testo Corretto (gold): {gold}

            Quale predizione è più simile al testo corretto?
            Rispondi esculivamente con una delle seguenti opzioni: A / B / TIE
            """
        else:
            compare_prompt = f"""

            Prediction A (Minerva): {minerva_pred}
            Prediction B (TinyLLaMA): {tinyllama_pred}
            Corrected Text (gold): {gold}

            Which prediction is most similar to the correct text?
            Answer only with one of the following options: A / B / TIE
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
