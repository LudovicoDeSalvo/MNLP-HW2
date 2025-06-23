# src/data_prep.py

import os
import json
import random
import math
import re
import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
import nltk
from src.utils import build_prompt

# Ensure NLTK is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK sentence tokenizer model 'punkt'...")
    nltk.download('punkt', quiet=True)


def load_full_documents_for_eval(config, dataset_key, paths):
    """
    FIXED: This function was added to resolve a crash in the evaluation and correlation menus.
    It loads only the evaluation portion (last 20%) of the documents.
    """
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
        return None

    eval_doc_examples = []
    
    doc_ids = sorted(set(original_docs) & set(cleaned_docs))
    # Determine the split point for training and evaluation
    eval_start_index = int(len(doc_ids) * 0.8)
    eval_ids = set(doc_ids[eval_start_index:])

    print(f"Loading {len(eval_ids)} full documents for evaluation...")
    for doc_id in doc_ids:
        if doc_id in eval_ids:
            eval_doc_examples.append({
                "id": doc_id,
                "noisy_doc": original_docs[doc_id].strip(),
                "target_doc": cleaned_docs[doc_id].strip()
            })
            
    return pd.DataFrame(eval_doc_examples)


def prepare_dataset(config, dataset_key, paths):
    """
    NOTE: This function is unchanged. Per your request, it will always process
    the dataset using sentence-splitting and will NOT use the Gemini-chunked file.
    """
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


def parse_gemini_split(response_text):
    """Parses the structured text output from Gemini to extract aligned chunks."""
    try:
        noisy_chunks = re.findall(r"NOISY_CHUNK_\d+:\s```(.*?)```", response_text, re.DOTALL)
        clean_chunks = re.findall(r"CLEAN_CHUNK_\d+:\s```(.*?)```", response_text, re.DOTALL)
        
        noisy_chunks = [chunk.strip() for chunk in noisy_chunks]
        clean_chunks = [chunk.strip() for chunk in clean_chunks]
        
        if len(noisy_chunks) == len(clean_chunks) and noisy_chunks:
            return noisy_chunks, clean_chunks
        else:
            print(f"Warning: Mismatch in parsed chunks. Noisy: {len(noisy_chunks)}, Clean: {len(clean_chunks)}")
            return None, None
    except Exception as e:
        print(f"Error parsing Gemini response: {e}")
        return None, None

def split_text_with_gemini(noisy_doc, clean_doc, num_chunks, gemini_model):
    """Uses Gemini with a specific prompt to split both texts into aligned chunks."""
    if num_chunks <= 1:
        return [noisy_doc], [clean_doc]
    prompt = f"""
    You are an expert text processor. Your task is to split two versions of a text (a noisy OCR version and a clean version) into {num_chunks} semantically coherent and roughly equal-sized sub-paragraphs. It is critical that the chunks are aligned, meaning NOISY_CHUNK_1 corresponds exactly to CLEAN_CHUNK_1.

    Here are the two versions of the text:

    --- OCR (NOISY) VERSION ---
    {noisy_doc}
    --- END OCR (NOISY) VERSION ---

    --- CLEAN VERSION ---
    {clean_doc}
    --- END CLEAN VERSION ---

    Split them into {num_chunks} aligned parts. Your output MUST follow this exact format, with each chunk enclosed in triple backticks:

    NOISY_CHUNK_1:
    ```
    (content of the first noisy chunk)
    ```
    CLEAN_CHUNK_1:
    ```
    (content of the first clean chunk)
    ```
    ...and so on for all {num_chunks} chunks.
    """
    try:
        response = gemini_model.generate_content(prompt)
        return parse_gemini_split(response.text)
    except Exception as e:
        print(f"Error calling Gemini API for splitting: {e}")
        return None, None

def create_chunked_dataset(config, dataset_key, paths, gemini_model):
    """
    Reads the raw dataset, uses Gemini to create intelligent chunks,
    and saves the result to a new file.
    NOTE: This function is unchanged.
    """
    print(f"\nStarting Gemini-powered chunking for the '{dataset_key}' dataset...")
    dataset_config = config["datasets"][dataset_key]
    dataset_dir = paths['dataset_dir']
    
    original_path = os.path.join(dataset_dir, dataset_key, dataset_config['original_filename'])
    cleaned_path = os.path.join(dataset_dir, dataset_key, dataset_config['cleaned_filename'])
    output_path = os.path.join(dataset_dir, dataset_key, "preprocessed_chunks.json")
    
    try:
        with open(original_path, "r", encoding="utf-8") as f:
            original_docs = json.load(f)
        with open(cleaned_path, "r", encoding="utf-8") as f:
            cleaned_docs = json.load(f)
    except FileNotFoundError as e:
        print(f"❌ Raw dataset file not found: {e}.")
        return

    TOKEN_LIMIT_PER_CHUNK = 1500
    all_examples = []
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    doc_ids = sorted(list(original_docs.keys()))
    for doc_id in tqdm(doc_ids, desc="Processing Documents"):
        if doc_id not in cleaned_docs:
            continue
            
        noisy_doc = original_docs[doc_id].strip()
        clean_doc = cleaned_docs[doc_id].strip()
        
        total_tokens = len(tokenizer.encode(noisy_doc))
        num_chunks = math.ceil(total_tokens / TOKEN_LIMIT_PER_CHUNK)
        
        if num_chunks > 1:
            noisy_chunks, clean_chunks = split_text_with_gemini(noisy_doc, clean_doc, num_chunks, gemini_model)
        else:
            noisy_chunks, clean_chunks = [noisy_doc], [clean_doc]
        
        if noisy_chunks:
            for i in range(len(noisy_chunks)):
                all_examples.append({
                    "id": f"{doc_id}_{i+1}",
                    "noisy": noisy_chunks[i],
                    "target": clean_chunks[i]
                })

    if not all_examples:
        print("❌ No examples were created.")
        return

    print(f"\nSaving {len(all_examples)} processed chunks to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_examples, f, ensure_ascii=False, indent=2)
    
    print("✅ Pre-processing complete!")