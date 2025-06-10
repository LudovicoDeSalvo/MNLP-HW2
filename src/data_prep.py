import random               
import re                  
import json                 
import pandas as pd         
import numpy as np         
from datasets import Dataset           
from sklearn.model_selection import train_test_split  
from sentence_transformers import SentenceTransformer  
from wtpsplit import SaT as SaTSplitter

from src.sat import align_sentences_by_semantics
from src.utils import build_prompt

#MAKE SURE TO PASS OCR_NOISE_PATTERNS
def inject_ocr_noise(OCR_NOISE_PATTERNS, text, num_errors=2):
    """Applies a number of random OCR-like errors to a string."""
    noisy = text
    for _ in range(num_errors):
        pattern, replacement = random.choice(OCR_NOISE_PATTERNS)
        noisy = re.sub(pattern, replacement, noisy, count=1)
    return noisy

def prepare_dataset(SENTENCE_SPLITTING, ITA, original_path, cleaned_path, test_size=0.2, random_state=42):
    """Loads, augments, and splits the dataset for training and evaluation."""
    with open(original_path, "r", encoding="utf-8") as f:
        original = json.load(f)
    with open(cleaned_path, "r", encoding="utf-8") as f:
        cleaned = json.load(f)

    if SENTENCE_SPLITTING:
        sentence_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    examples = []
    for k in sorted(set(original) & set(cleaned)):
        noisy_para = original[k].strip()
        clean_para = cleaned[k].strip()

        if SENTENCE_SPLITTING:   
            noisy_sents = SaTSplitter.split(noisy_para) 
            clean_sents = SaTSplitter.split(clean_para)
            aligned = align_sentences_by_semantics(noisy_sents, clean_sents, sentence_model)

            for noisy, clean in aligned:
                prompt = build_prompt(ITA , noisy)
                full_text = f"{prompt} {clean}"
                examples.append({"text": full_text, "source": "sat-segmented"})
        else:
            prompt = build_prompt(ITA , noisy_para)
            full_text = f"{prompt} {clean_para}"

            examples.append({"text": full_text, "source": "no-split"})
            
    random.shuffle(examples)
    train_data, eval_data = train_test_split(examples, test_size=test_size, random_state=random_state)
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    print(f"📚 Dataset prepared: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset