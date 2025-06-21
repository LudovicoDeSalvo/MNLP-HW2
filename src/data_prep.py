import json
import random
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from src.utils import build_prompt

def prepare_dataset(config, dataset_key, test_size=0.2, random_state=42):
    """Loads and splits the dataset using paths and settings from the config file."""
    
    dataset_config = config["datasets"][dataset_key]
    
    try:
        with open(dataset_config["original_path"], "r", encoding="utf-8") as f:
            original = json.load(f)
        with open(dataset_config["cleaned_path"], "r", encoding="utf-8") as f:
            cleaned = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Dataset file not found: {e}. Please check paths in your config for the '{dataset_key}' dataset.")
        return None, None

    examples = []
    for k in sorted(set(original) & set(cleaned)):
        noisy_para = original[k].strip()
        clean_para = cleaned[k].strip()
        
        examples.append({
            "id": k,
            "noisy": noisy_para,
            "prompt": build_prompt(noisy_para, config["prompt_style"], dataset_config.get("ita_language", False)),
            "target": clean_para,
        })
            
    random.shuffle(examples)
    train_data, eval_data = train_test_split(examples, test_size=test_size, random_state=random_state)
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    eval_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
    
    print(f"📚 Dataset '{dataset_key}' prepared: {len(train_dataset)} training samples, {len(eval_dataset)} evaluation samples.")
    return train_dataset, eval_dataset