import os
import random
import json
import numpy as np
import torch
import google.generativeai as genai
from huggingface_hub import login
from transformers import set_seed
import Levenshtein

def load_config(config_path):
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"✅ Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Error: Configuration file not found at {config_path}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Error: Could not decode JSON from {config_path}")
        return None

def build_prompt(noisy_text, prompt_style, ita=False):
    """Builds a model-specific prompt based on the style from the config."""
    if ita:
        # For Italian, we'll use a specific, structured prompt for TinyLlama
        if prompt_style == "tinyllama":
            return f"""Sei un assistente esperto nella correzione di testi OCR. Il tuo compito è correggere il testo OCR fornito, sistemando ogni errore, refuso o problema di formattazione. Restituisci solo il testo corretto, senza commenti o spiegazioni aggiuntive.

                    ### TESTO OCR:
                    {noisy_text}

                    ### TESTO CORRETTO:"""
        # The Minerva prompt for Italian can remain the same if it works well
        elif prompt_style == "minerva":
            return f"Correggi il seguente testo OCR:\n\n{noisy_text}\n\nTesto corretto:"

    # --- English Prompts ---
    if prompt_style == "tinyllama":
        # A more structured prompt for English as well
        return f"""You are an expert OCR correction assistant. Your task is to correct the given OCR text, fixing any errors, misspellings, or formatting issues. Return only the perfectly corrected text, without any additional comments or explanations.

                ### OCR TEXT:
                {noisy_text}

                ### CORRECTED TEXT:"""
    elif prompt_style == "minerva":
        return f"""### SYSTEM
                You are a careful OCR fixer. Given a noisy paragraph, return only the corrected version.

                ### USER
                <<<
                {noisy_text}
                >>>

                ### RESPONSE
                """
    else:
        # A default fallback prompt
        return f"Correct the following OCR text:\n\n{noisy_text}\n\nCorrected text:"

def set_all_seeds(seed=42):
    """Sets seeds for all relevant libraries to ensure reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def login_to_huggingface(paths):
    token_path = os.path.join(paths['general_utils_dir'], 'hf_token.txt')

    try:
        with open(token_path) as f:
            token = f.read().strip()
            login(token=token, add_to_git_credential=True)
            print("✅ Successfully logged into HuggingFace Hub.")
    except Exception as e:
        print(f"⚠️ HuggingFace login failed: {e}.")

def configure_gemini(paths):
    api_key_path = os.path.join(paths['general_utils_dir'], 'google_api.txt')
    
    try:
        with open(api_key_path) as f:
            api_key = f.read().strip()
            genai.configure(api_key=api_key)
            print("✅ Successfully configured Gemini API.")
            return genai.GenerativeModel("gemini-1.5-flash-latest")
    except Exception as e:
        print(f"⚠️ Failed to configure Gemini API: {e}. Evaluation with Gemini will fail.")
        return None

def calculate_cer(s1, s2):
    """Calculates the Character Error Rate (CER) between two strings."""
    s1 = s1.replace(' ', '')
    s2 = s2.replace(' ', '')
    if not s2:
        return 1.0 if s1 else 0.0
    return Levenshtein.distance(s1, s2) / len(s2)