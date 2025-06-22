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
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt', quiet=True)


from src.utils import build_prompt, calculate_cer

EVAL_RESULTS_DIR = "evaluation_results"

# This function is now passed the 'ita' flag from the dataset config
def gemini_judge_score(noisy, predicted, gold, gemini_model, ita=False):
    if not gemini_model:
        return -1

    prompt_eng = f"""
    You are an expert judge of text quality. This is CORRECTED OCR text. Note any mistakes in spelling, grammar, punctuation, or formatting. Check semantinc logic, context consistency and possible hallucinations.

    Here is the text:

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

    prompt_ita =  f"""
    Sei un giudice esperto della qualità del testo. Verrà fornito un test OCR CORRETTO, da confrontare col il testo di rifermento (100% Corretto).
    Devi essere sensibile a errori di ortografia, grammatica, punteggiatura o formattazione.
    Controlla la logica semantica, la coerenza contestuale e possibili allucinazioni.
    Devi assegnare un voto da 0 a 10, dando i seguenti punteggi ad ogni categoria:
    Punteggio da 0 a 4 per Leggibilità generale: quanto è facile e scorrevole leggere il testo.
    Punteggio da 0 a 2 per Correttezza: errori ortografici, punteggiatura, typos.
    Punteggio da 0 a 1 per Formattazione: corretta spaziatura e interruzioni di riga.
    Punteggio da 0 a 3 per Coerenza semantica: le frasi hanno senso.

    Ecco il testo:

    "{predicted}"

    FINE TESTO
    Testo di riferimento:

    "{gold}"

    FINE TESTO DI RIFERMENTO

    Comapara i due testi e fornisci un giudizio. La tua intera risposta deve essere un singolo numero da 0 a 10 ovvero la somma dei punteggi delle singole categorie.
    """

    prompt = prompt_ita if ita else prompt_eng

    try:
        response = gemini_model.generate_content(prompt)
        match = re.search(r'\d+', response.text)
        return int(match.group(0)) if match else -1
    except Exception as e:
        print(f"Error during Gemini scoring: {e}")
        return -1

# In src/eval.py

def correct_document(doc_text, model, tokenizer, config, dataset_config):
    """
    Segments long documents, corrects each chunk with a two-pass system,
    and now includes detailed print statements for progress tracking.
    """
    is_ita = dataset_config.get("ita_language", False)
    prompt_style = config["prompt_style"]
    device = model.device

    # --- 1. Robust Chunking Logic ---
    correction_prompt_template = build_prompt("{text_placeholder}", prompt_style, ita=is_ita, pass_type="correction")
    prompt_token_count = len(tokenizer.encode(correction_prompt_template.replace("{text_placeholder}", "")))
    CONTENT_TOKEN_LIMIT = 2048 - prompt_token_count - 50

    sentences = nltk.sent_tokenize(doc_text, language='italian' if is_ita else 'english')
    final_chunks = []
    for sent in sentences:
        if len(tokenizer.encode(sent)) > CONTENT_TOKEN_LIMIT:
            words = sent.split()
            current_sub_chunk = ""
            for word in words:
                if len(tokenizer.encode(current_sub_chunk + " " + word)) > CONTENT_TOKEN_LIMIT:
                    final_chunks.append(current_sub_chunk.strip())
                    current_sub_chunk = word
                else:
                    current_sub_chunk += " " + word
            if current_sub_chunk:
                final_chunks.append(current_sub_chunk.strip())
        else:
            final_chunks.append(sent)

    # --- NEW: Announce the number of chunks ---
    print(f"\n  - Document split into {len(final_chunks)} chunks. Starting two-pass correction...")

    # --- 2. Two-Pass Correction with Inner Printing ---
    final_corrected_chunks = []
    for i, chunk in enumerate(final_chunks):
        # --- NEW: Print progress for each chunk ---
        print(f"    - Processing chunk {i + 1}/{len(final_chunks)}...")
        if not chunk.strip():
            continue
        
        # Pass 1
        print(f"      - Pass 1 (Correction)...", end="", flush=True)
        correction_prompt = build_prompt(chunk, prompt_style, ita=is_ita, pass_type="correction")
        inputs = tokenizer(correction_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=2048, num_beams=3, repetition_penalty=1.2)
        corrected_text_pass1 = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(" Done.")

        # Pass 2
        print(f"      - Pass 2 (Polishing)...", end="", flush=True)
        polishing_prompt = build_prompt(corrected_text_pass1, prompt_style, ita=is_ita, pass_type="polishing")
        inputs = tokenizer(polishing_prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=2048, num_beams=3, repetition_penalty=1.1)
        final_text = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        final_text = final_text.replace("<|system|>", "").replace("<|user|>", "").strip()
        print(" Done.")
        
        final_corrected_chunks.append(final_text)

    # --- 3. Reassembly ---
    print("  - Document correction complete.")
    return "\n".join(final_corrected_chunks)


def evaluate_model(config, dataset_key, eval_docs_df, paths, gemini_model, use_gemini_scoring):
    """Evaluates a single fine-tuned model by processing full documents chunk by chunk."""
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_name = config["model_name"]
    dataset_config = config["datasets"][dataset_key]
    is_ita = dataset_config.get("ita_language", False)

    model_path = os.path.join(paths['trained_models_dir'], config['output_dir_name'])
    
    print(f"\n====== Evaluating {model_name} from {model_path} on '{dataset_key}' dataset ======")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    except OSError:
        print(f"❌ Model not found at {model_path}. Please train it first.")
        return None
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for _, row in tqdm(eval_docs_df.iterrows(), total=len(eval_docs_df), desc=f"Evaluating docs for {model_name.split('/')[-1]}"):
        noisy_doc = row["noisy_doc"]
        target_doc = row["target_doc"]

        predicted_doc = correct_document(noisy_doc, model, tokenizer, config, dataset_config)
        
        gem_score = -1
        if use_gemini_scoring and gemini_model:
            gem_score = gemini_judge_score(noisy_doc, predicted_doc, target_doc, gemini_model, ita=is_ita)

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