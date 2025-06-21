import json
import pandas as pd
from datasets import Dataset
import nltk
import os
from src.utils import build_prompt

# Download the sentence tokenizer model from NLTK (only needs to be done once)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK sentence tokenizer model 'punkt'...")
    nltk.download('punkt', quiet=True)

def prepare_dataset(config, dataset_key, paths):
    dataset_config = config["datasets"][dataset_key]
    dataset_dir = paths['dataset_dir']

    original_path = os.path.join(dataset_dir, dataset_key, dataset_config['original_filename'])
    cleaned_path = os.path.join(dataset_dir, dataset_key, dataset_config['cleaned_filename'])
    
    try:
        with open(original_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_docs = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Dataset file not found: {e}. Please check paths in your config.")
        return None, None, None

    train_examples = []
    eval_sentence_examples = []
    eval_doc_examples = []
    
    doc_ids = sorted(set(original_docs) & set(cleaned_docs))
    train_ids = set(doc_ids[:int(len(doc_ids) * 0.8)])

    print("Processing documents into training and evaluation sets...")
    for doc_id in doc_ids:
        is_training_doc = doc_id in train_ids
        
        noisy_doc = original_docs[doc_id].strip()
        clean_doc = cleaned_docs[doc_id].strip()
        
        # Add to the full document evaluation set if it's an eval doc
        if not is_training_doc:
            eval_doc_examples.append({
                "id": doc_id,
                "noisy_doc": noisy_doc,
                "target_doc": clean_doc
            })

        # Process into sentence examples for training and trainer-evaluation
        noisy_sentences = nltk.sent_tokenize(noisy_doc, language='italian' if dataset_config.get("ita_language") else 'english')
        clean_sentences = nltk.sent_tokenize(clean_doc, language='italian' if dataset_config.get("ita_language") else 'english')
        num_sentences = min(len(noisy_sentences), len(clean_sentences))

        for i in range(num_sentences):
            noisy_sent = noisy_sentences[i].strip()
            clean_sent = clean_sentences[i].strip()
            
            if not noisy_sent or not clean_sent:
                continue

            example = {
                "id": f"{doc_id}_{i}",
                "noisy": noisy_sent,
                "prompt": build_prompt(noisy_sent, config["prompt_style"], dataset_config.get("ita_language", False)),
                "target": clean_sent,
            }
            if is_training_doc:
                train_examples.append(example)
            else:
                eval_sentence_examples.append(example)
    
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_examples))
    eval_sentence_dataset = Dataset.from_pandas(pd.DataFrame(eval_sentence_examples))
    eval_docs_df = pd.DataFrame(eval_doc_examples)

    print(f"📚 Dataset prepared: {len(train_dataset)} training sentences, {len(eval_sentence_dataset)} eval sentences, {len(eval_docs_df)} eval docs.")
    return train_dataset, eval_sentence_dataset, eval_docs_df