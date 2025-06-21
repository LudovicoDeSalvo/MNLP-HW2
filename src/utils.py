import os           
import random        
import re            
import numpy as np   
import torch         
import google.generativeai as genai  
from huggingface_hub import login    
from transformers import set_seed    
import json
import pandas as pd
import Levenshtein


def build_prompt(ITA, noisy_text, model_name=None):
    
    if ITA:
        mario = 0 # PLACEHOLDER
    else:
        if "TinyLlama" in model_name:
            return f"""Correct the following OCR text:

                {noisy_text}

                Corrected text:"""
            
        else:  # Minerva or other base models
            return f"""### SYSTEM
                    You are a careful OCR fixer. Given a noisy paragraph, return only the corrected version.

                    ### USER
                    <<<
                    {noisy_text}
                    >>>

                    ### RESPONSE
                    """

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
    

def calculate_cer(s1, s2):
    """
    Calculates the Character Error Rate (CER) between two strings.
    """
    s1 = s1.replace(' ', '')
    s2 = s2.replace(' ', '')
    distance = Levenshtein.distance(s1, s2)
    length = len(s2)
    if length == 0:
        return 1.0 if distance > 0 else 0.0
    return distance / length