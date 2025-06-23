# main.py

import os
import sys
import subprocess
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import nltk
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# --- DYNAMIC PATH DETECTION ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.path.abspath(".")
PATHS = {
    "root": PROJECT_ROOT, "dataset_dir": os.path.join(PROJECT_ROOT, "dataset"),
    "general_utils_dir": os.path.join(PROJECT_ROOT, "general_utils"),
    "model_configs_dir": PROJECT_ROOT, "evaluation_results_dir": os.path.join(PROJECT_ROOT, "evaluation_results"),
    "trained_models_dir": os.path.join(PROJECT_ROOT, "trained_models")
}
sys.path.insert(0, PATHS["root"])
# --- END DYNAMIC PATH DETECTION ---

# Corrected and consolidated imports
from src.utils import set_all_seeds, login_to_huggingface, configure_gemini, load_config, build_prompt
from src.train import train_model
from src.eval import evaluate_model, gemini_judge_score
# FIXED: This import now works because the function has been added to data_prep.py
from src.data_prep import create_chunked_dataset, prepare_dataset, load_full_documents_for_eval

# --- GLOBAL VARIABLES ---
MODEL_CONFIGS = {}
GEMINI_MODEL = None

def get_model_configs():
    """Dynamically finds all model config files."""
    configs = {}
    config_dir = PATHS["model_configs_dir"]
    try:
        files = sorted([f for f in os.listdir(config_dir) if f.startswith('config_') and f.endswith('.json')])
        for i, f in enumerate(files):
            model_name = f.replace('config_', '').replace('.json', '').capitalize()
            configs[str(i + 1)] = {"name": model_name, "path": os.path.join(config_dir, f)}
    except FileNotFoundError:
        print(f"Warning: Could not find model config directory: {config_dir}")
    return configs

# --- HELPER FUNCTIONS FOR MENU ---
def ask_yes_no(question):
    """Simple helper to ask a yes/no question."""
    while True:
        response = input(f"{question} (y/n): ").lower().strip()
        if response in ['y', 'yes']: return True
        if response in ['n', 'no']: return False
        print("Invalid input. Please enter 'y' or 'n'.")

def select_dataset():
    """Lets the user select a dataset to work with."""
    print("\nWhich dataset do you want to use?")
    print("1. English")
    print("2. Italian")
    while True:
        choice = input("Enter your choice (1-2): ")
        if choice == "1": return "eng"
        if choice == "2": return "ita"
        print("Invalid choice. Please enter 1 or 2.")

def select_model_configs_from_menu(all_option=True):
    """Lets the user select one or more model configurations from a generated menu."""
    if not MODEL_CONFIGS:
        print("❌ No model configuration files found in the root directory.")
        return None
    
    print("\nSelect model(s):")
    for key, val in MODEL_CONFIGS.items():
        print(f"{key}. {val['name']}")
    
    if all_option:
        print(f"{len(MODEL_CONFIGS) + 1}. All")
    
    while True:
        choice_limit = len(MODEL_CONFIGS) + 1 if all_option else len(MODEL_CONFIGS)
        choice = input(f"Enter your choice (1-{choice_limit}): ")
        if choice in MODEL_CONFIGS:
            return [load_config(MODEL_CONFIGS[choice]["path"])]
        if all_option and choice == str(len(MODEL_CONFIGS) + 1):
            return [load_config(val["path"]) for val in MODEL_CONFIGS.values()]
        print("Invalid choice.")

# --- MENU HANDLERS ---

def handle_train_model():
    """Handler for training models using the standard data preparation."""
    dataset_key = select_dataset()
    selected_configs = select_model_configs_from_menu()
    if not selected_configs: return
    
    # FIXED: Removed the confusing and now-irrelevant question about sentence splitting.
    # The standard prepare_dataset function will always be used.
    print("\nPreparing data using standard sentence splitting method...")
    
    # FIXED: Corrected the function call to match its definition (removed extra argument)
    # and correctly unpacked the three return values, ignoring the third.
    train_ds, eval_ds, _ = prepare_dataset(selected_configs[0], dataset_key, PATHS)

    if train_ds is None or eval_ds is None:
        print("❌ Could not prepare dataset.")
        return

    for config in selected_configs:
        if not config: continue
        train_model(config, PATHS, train_ds, eval_ds)

def handle_evaluate_model():
    """Handler for evaluating models on full documents."""
    dataset_key = select_dataset()
    selected_configs = select_model_configs_from_menu()
    if not selected_configs: return

    print(f"\n--- Loading full documents for evaluation on '{dataset_key}' dataset ---")
    eval_docs_df = load_full_documents_for_eval(selected_configs[0], dataset_key, PATHS)
    
    if eval_docs_df is None or eval_docs_df.empty:
        print("❌ Could not load documents for evaluation.")
        return

    use_gemini = ask_yes_no("⭐ Use Gemini for scoring?")
    if use_gemini and not GEMINI_MODEL:
        print("\n❌ Cannot use Gemini scoring. The initial API connection failed.")
        use_gemini = False

    for config in selected_configs:
        if not config: continue
        results_df = evaluate_model(config, dataset_key, eval_docs_df, PATHS, GEMINI_MODEL, use_gemini)
        
        if results_df is not None and not results_df.empty:
            avg_levenshtein = results_df['levenshtein'].mean()
            avg_cer = results_df['char_error_rate'].mean()
            model_name = config.get('model_name', 'Unknown Model')
            
            print("\n" + "-"*20 + " SUMMARY " + "-"*20)
            print(f"Model: {model_name}")
            print(f"📊 Average Levenshtein Score: {avg_levenshtein:.4f}")
            print(f"📊 Average Character Error Rate (CER): {avg_cer:.4f}")

            if use_gemini:
                valid_gemini_scores = results_df[results_df['gemini_score'] != -1]['gemini_score']
                if not valid_gemini_scores.empty:
                    avg_gemini = valid_gemini_scores.mean()
                    print(f"✨ Average Gemini Score: {avg_gemini:.4f}")
            print("-"*(49))

def handle_gemini_preprocess():
    """Handler for the optional, one-time Gemini-powered data chunking."""
    global GEMINI_MODEL
    print("\n--- (Experimental) Pre-process Dataset with Gemini ---")
    if not GEMINI_MODEL:
        print("Error: This feature requires a Gemini API connection.")
        return
        
    dataset_key = select_dataset()
    config = load_config(list(MODEL_CONFIGS.values())[0]['path'])
    if not config: return

    output_path = os.path.join(PATHS['dataset_dir'], dataset_key, "preprocessed_chunks.json")
    if os.path.exists(output_path):
        print(f"\n✅ Pre-processed file already exists at: {output_path}")
        if not ask_yes_no("Do you want to overwrite it?"):
            print("Pre-processing cancelled.")
            return

    print("\nThis process will make API calls to Gemini to create 'preprocessed_chunks.json'.")
    print("NOTE: Per your request, this file will NOT be used for training.")
    if ask_yes_no("Do you want to continue?"):
        create_chunked_dataset(config, dataset_key, PATHS, GEMINI_MODEL)

def handle_human_correlation():
    """Interactive workflow to compare user scores with Gemini scores."""
    global GEMINI_MODEL
    print("\n--- Interactive Human vs. Gemini Correlation ---")

    NUM_SENTENCES_PER_SAMPLE = 3
    print(f"(Each sample will consist of the first {NUM_SENTENCES_PER_SAMPLE} sentences of a random paragraph)")

    if not GEMINI_MODEL:
        print("Re-establishing Gemini connection for this session...")
        GEMINI_MODEL = configure_gemini(PATHS)
        if not GEMINI_MODEL:
            print("❌ Gemini API connection is not available.")
            return

    dataset_key = select_dataset()
    print("\nSelect the model to evaluate:")
    selected_configs = select_model_configs_from_menu(all_option=False)
    if not selected_configs: return
    config = selected_configs[0]

    num_samples = 0
    while num_samples <= 0:
        try:
            num_samples = int(input(f"How many random paragraphs to sample? (e.g., 5): "))
            if num_samples <= 0: print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    print("\nLoading and preparing random paragraph samples...")
    eval_docs_df = load_full_documents_for_eval(config, dataset_key, PATHS)
    
    if eval_docs_df is None or eval_docs_df.empty:
        print("❌ Could not find any evaluation documents to sample from.")
        return

    if len(eval_docs_df) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(eval_docs_df)} available.")
        num_samples = len(eval_docs_df)
    
    random_docs = eval_docs_df.sample(n=num_samples, random_state=42)
    
    random_samples = []
    is_ita_lang = config["datasets"][dataset_key].get("ita_language", False)
    
    for _, doc_row in random_docs.iterrows():
        noisy_sentences = nltk.sent_tokenize(doc_row['noisy_doc'], language='italian' if is_ita_lang else 'english')
        target_sentences = nltk.sent_tokenize(doc_row['target_doc'], language='italian' if is_ita_lang else 'english')
        
        noisy_snippet = " ".join(noisy_sentences[:NUM_SENTENCES_PER_SAMPLE])
        target_snippet = " ".join(target_sentences[:NUM_SENTENCES_PER_SAMPLE])
        
        random_samples.append({"noisy": noisy_snippet, "target": target_snippet})

    print(f"Loading model '{config['model_name']}'...")
    try:
        model_path = os.path.join(PATHS['trained_models_dir'], config['output_dir_name'])
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"❌ Could not load model from {model_path}. Please train it first.")
        return
    
    user_scores, gemini_scores = [], []
    dataset_config = config["datasets"][dataset_key]
    is_ita = dataset_config.get("ita_language", False)

    for i, sample in enumerate(random_samples):
        print("\n" + "="*50 + f"\n--- Sample {i + 1}/{num_samples} ---")
        ocr_text, target_text = sample['noisy'], sample['target']
        
        print("\n1. Getting model correction...")
        prompt = build_prompt(ocr_text, config["prompt_style"], is_ita)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids=inputs.input_ids, max_new_tokens=2048, repetition_penalty=1.2, num_beams=3)
        predicted_text = tokenizer.decode(output_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        predicted_text = predicted_text.replace("<|system|>", "").replace("<|user|>", "").strip()
        
        print("2. Getting Gemini score...")
        gemini_score = gemini_judge_score(ocr_text, predicted_text, target_text, GEMINI_MODEL, ita=is_ita)

        print("\n--- PLEASE EVALUATE THE CORRECTION ---")
        print(f"\n[ORIGINAL OCR SNIPPET]:\n{ocr_text}")
        print(f"\n[MODEL CORRECTION]:\n{predicted_text}")
        print(f"\n[GROUND TRUTH SNIPPET]:\n{target_text}")
        print("-" * 20 + f"\nGemini's Score: {gemini_score}\n" + "-" * 20)
        
        score_min, score_max = (0, 30) if is_ita else (1, 5)
        user_score = -1
        while not (score_min <= user_score <= score_max):
            try:
                score_input = input(f"Enter your score ({score_min}-{score_max}): ")
                user_score = int(score_input)
                if not (score_min <= user_score <= score_max):
                    print(f"Invalid score.")
            except ValueError:
                print("Invalid input.")
        user_scores.append(user_score); gemini_scores.append(gemini_score)

    print("\n" + "="*50 + "\n--- Correlation Analysis Complete ---")
    if len(user_scores) > 1:
        kappa_labels = list(range(0, 31)) if is_ita else list(range(1, 6))
        pearson_corr = pd.Series(user_scores).corr(pd.Series(gemini_scores))
        mae = mean_absolute_error(user_scores, gemini_scores)
        try:
            kappa = cohen_kappa_score(user_scores, gemini_scores, labels=kappa_labels)
        except ValueError:
            kappa = float('nan')
        print(f"Number of samples evaluated: {len(user_scores)}")
        print(f"\n📊 Pearson Correlation: {pearson_corr:.4f}")
        print(f"📊 Mean Absolute Error (MAE): {mae:.4f}")
        print(f"📊 Cohen's Kappa: {kappa:.4f}")
    else:
        print("Not enough samples to calculate correlation.")
    print("="*50)

def handle_install_requirements():
    """Installs packages from requirements.txt."""
    print("🔧 Installing/updating required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully.")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"❌ An error occurred during installation: {e}")

def main_menu():
    global GEMINI_MODEL, MODEL_CONFIGS
    if not PATHS: return
    
    MODEL_CONFIGS = get_model_configs()
    
    print("\nAttempting to connect to Gemini API...")
    GEMINI_MODEL = configure_gemini(PATHS)
    if not GEMINI_MODEL:
        print("⚠️ Gemini connection failed. Gemini-dependent features will be unavailable.")
    
    set_all_seeds(42)
    login_to_huggingface(PATHS)
    
    while True:
        print("\n==============================")
        print("   OCR Post-Correction Menu")
        print("==============================")
        print("1. Train Model (Standard)")
        print("2. Evaluate Model (Standard)")
        print("3. Human vs. Gemini Correlation")
        print("---")
        print("4. (Experimental) Create Gemini-Chunked Dataset")
        print("5. Install/Update Requirements")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")

        if choice == "1": handle_train_model()
        elif choice == "2": handle_evaluate_model()
        elif choice == "3": handle_human_correlation()
        elif choice == "4": handle_gemini_preprocess()
        elif choice == "5": handle_install_requirements()
        elif choice == "6":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice.")

if __name__ == "__main__":
    main_menu()