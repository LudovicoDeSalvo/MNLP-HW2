import os
import sys
import subprocess
import pandas as pd
from datasets import Dataset
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# --- DYNAMIC PATH DETECTION (NO CONFIG FILE NEEDED) ---
try:
    # The absolute path to the directory containing main.py
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments like Jupyter/Colab notebooks
    PROJECT_ROOT = os.path.abspath(".")

PATHS = {
    "root": PROJECT_ROOT,
    "dataset_dir": os.path.join(PROJECT_ROOT, "dataset"),
    "general_utils_dir": os.path.join(PROJECT_ROOT, "general_utils"),
    "model_configs_dir": PROJECT_ROOT,  # Assumes config files are in the root
    "evaluation_results_dir": os.path.join(PROJECT_ROOT, "evaluation_results"),
    "trained_models_dir": os.path.join(PROJECT_ROOT, "trained_models")
}
# This allows scripts in /src to be imported correctly
sys.path.insert(0, os.path.join(PATHS["root"], "src"))
# --- END DYNAMIC PATH DETECTION ---


# Now that the path is set, we can import our modules
from utils import set_all_seeds, login_to_huggingface, configure_gemini, load_config
from data_prep import prepare_dataset
from train import train_model
from eval import evaluate_model, get_single_correction, gemini_judge_score

def get_model_configs():
    """Dynamically finds all model config files from the specified directory."""
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

MODEL_CONFIGS = get_model_configs()
GEMINI_MODEL = None

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

def select_model_configs_from_menu():
    """Lets the user select one or more model configurations from a generated menu."""
    if not MODEL_CONFIGS:
        print("❌ No model configuration files found in the root directory (e.g., config_tinyllama.json).")
        return None
    
    print("\nSelect model(s):")
    for key, val in MODEL_CONFIGS.items():
        print(f"{key}. {val['name']}")
    print(f"{len(MODEL_CONFIGS) + 1}. All")
    
    while True:
        choice = input(f"Enter your choice (1-{len(MODEL_CONFIGS) + 1}): ")
        if choice in MODEL_CONFIGS:
            return [load_config(MODEL_CONFIGS[choice]["path"])]
        if choice == str(len(MODEL_CONFIGS) + 1):
            return [load_config(val["path"]) for val in MODEL_CONFIGS.values()]
        print("Invalid choice.")

def select_evaluation_file():
    """Lets the user select a single evaluation file to analyze."""
    # FIXED: Use the 'PATHS' dictionary instead of the old undefined constant
    eval_dir = PATHS['evaluation_results_dir']
    
    if not os.path.exists(eval_dir):
        print(f"❌ The '{eval_dir}' directory does not exist. Run an evaluation first.")
        return None, None
    
    try:
        eval_files = sorted([f for f in os.listdir(eval_dir) if f.endswith('.csv')])
        if not eval_files:
            print(f"❌ No evaluation files found in the '{eval_dir}' directory.")
            return None, None
    except FileNotFoundError:
        print(f"❌ The '{eval_dir}' directory does not exist. Run an evaluation first.")
        return None, None
    
    print("\nPlease select an evaluation file to analyze:")
    for i, f in enumerate(eval_files):
        print(f"  {i+1}: {f}")
    
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(eval_files)}): ")) - 1
            if 0 <= choice < len(eval_files):
                filename = eval_files[choice]
                dataset_key = 'ita' if '_ita.csv' in filename else 'eng'
                # FIXED: Use the correct variable for the path
                return os.path.join(eval_dir, filename), dataset_key
            print("Invalid number.")
        except (ValueError, IndexError):
            print("Please enter a valid number.")

# --- MENU HANDLERS ---
def handle_train_model():
    dataset_key = select_dataset()
    selected_configs = select_model_configs_from_menu()
    if not selected_configs: return

    print(f"\n--- Preparing data on '{dataset_key}' dataset ---")
    master_config = selected_configs[0]
    if not master_config: return
    
    train_ds, eval_sentence_ds, _ = prepare_dataset(master_config, dataset_key, PATHS)
    
    if train_ds is None or eval_sentence_ds is None:
        print("Could not prepare datasets. Aborting training.")
        return

    for config in selected_configs:
        if not config: continue
        train_model(config, PATHS, train_ds, eval_sentence_ds)

def handle_evaluate_model():
    global GEMINI_MODEL
    dataset_key = select_dataset()
    selected_configs = select_model_configs_from_menu()
    if not selected_configs: return

    print(f"\n--- Preparing data for evaluation on '{dataset_key}' dataset ---")
    master_config = selected_configs[0]
    if not master_config: return

    _, _, eval_docs_df = prepare_dataset(master_config, dataset_key, PATHS)
    
    if eval_docs_df is None or eval_docs_df.empty:
        print("Could not prepare evaluation documents. Aborting.")
        return

    use_gemini = ask_yes_no("⭐ Use Gemini for scoring? (Requires API key)")
    if use_gemini and not GEMINI_MODEL:
        GEMINI_MODEL = configure_gemini(PATHS)
        if not GEMINI_MODEL:
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

def handle_human_correlation():
    """Interactive workflow to compare user scores with Gemini scores."""
    global GEMINI_MODEL
    print("\n--- Interactive Human vs. Gemini Correlation ---")

    # 1. SETUP
    if not GEMINI_MODEL:
        print("❌ Gemini API connection is not available. This feature cannot be used.")
        return

    dataset_key = select_dataset()
    
    print("\nSelect the model to evaluate:")
    # We only evaluate one model at a time in this mode
    model_configs_list = [v for k, v in MODEL_CONFIGS.items()]
    for i, model_info in enumerate(model_configs_list):
        print(f"{i+1}. {model_info['name']}")
    
    model_choice = -1
    while model_choice < 0 or model_choice >= len(model_configs_list):
        try:
            choice = int(input(f"Enter choice (1-{len(model_configs_list)}): ")) - 1
            if 0 <= choice < len(model_configs_list):
                model_choice = choice
            else:
                print("Invalid number.")
        except ValueError:
            print("Please enter a number.")

    config = load_config(model_configs_list[model_choice]["path"])
    if not config: return
    
    num_samples = 0
    while num_samples <= 0:
        try:
            num_samples = int(input("How many random samples to evaluate? (e.g., 5): "))
            if num_samples <= 0: print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")

    # 2. DATA PREPARATION
    print("\nLoading and preparing random samples...")
    train_ds, eval_sentence_ds, _ = prepare_dataset(config, dataset_key, PATHS)
    full_ds = pd.concat([train_ds.to_pandas(), eval_sentence_ds.to_pandas()])
    
    if len(full_ds) < num_samples:
        print(f"Warning: Requested {num_samples} samples, but only {len(full_ds)} are available.")
        num_samples = len(full_ds)
        
    random_samples = full_ds.sample(n=num_samples).to_dict('records')

    # 3. LOAD MODEL
    print(f"Loading model '{config['model_name']}'...")
    try:
        model_path = os.path.join(PATHS['trained_models_dir'], config['output_dir_name'])
        model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except OSError:
        print(f"❌ Could not load model from {model_path}. Please train it first.")
        return
    
    # 4. INTERACTIVE LOOP
    user_scores = []
    gemini_scores = []
    dataset_config = config["datasets"][dataset_key]
    is_ita = dataset_config.get("ita_language", False)

    for i, sample in enumerate(random_samples):
        print("\n" + "="*50)
        print(f"--- Sample {i + 1}/{num_samples} ---")
        
        ocr_text = sample['noisy']
        target_text = sample['target']
        
        print("\n1. Getting model correction...")
        predicted_text = get_single_correction(ocr_text, model, tokenizer, config, dataset_config)
        
        print("2. Getting Gemini score...")
        gemini_score = gemini_judge_score(ocr_text, predicted_text, target_text, GEMINI_MODEL, ita=is_ita)

        print("\n--- PLEASE EVALUATE THE CORRECTION ---")
        print(f"\n[ORIGINAL OCR]:\n{ocr_text}")
        print(f"\n[MODEL CORRECTION]:\n{predicted_text}")
        print(f"\n[GROUND TRUTH]:\n{target_text}")
        print("-" * 20)
        print(f"Gemini's Score: {gemini_score}")
        print("-" * 20)
        
        # Get user score with input validation
        user_score = -1
        while user_score < 0 or user_score > 5:
            try:
                score_input = input("Enter your score (0-5): ")
                user_score = int(score_input)
                if not (0 <= user_score <= 5):
                    print("Invalid score. Please enter a number between 0 and 5.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        
        user_scores.append(user_score)
        gemini_scores.append(gemini_score)

    # 5. FINAL ANALYSIS
    print("\n" + "="*50)
    print("--- Correlation Analysis Complete ---")
    if len(user_scores) > 1:
        pearson_corr = pd.Series(user_scores).corr(pd.Series(gemini_scores))
        mae = mean_absolute_error(user_scores, gemini_scores)
        kappa = cohen_kappa_score(user_scores, gemini_scores, labels=[0,1,2,3,4,5])

        print(f"Number of samples evaluated: {len(user_scores)}")
        print(f"\n📊 Pearson Correlation: {pearson_corr:.4f}")
        print(f"   (Measures linear relationship. +1 is perfect positive correlation.)")
        print(f"\n📊 Mean Absolute Error (MAE): {mae:.4f}")
        print(f"   (Average difference between your scores and Gemini's. Lower is better.)")
        print(f"\n📊 Cohen's Kappa: {kappa:.4f}")
        print(f"   (Measures agreement vs. chance. >0.6 is substantial, >0.8 is almost perfect.)")
    else:
        print("Not enough samples to calculate correlation. Please evaluate at least 2 samples.")
    print("="*50)

def handle_install_requirements():
    """Installs packages from requirements.txt."""
    print("🔧 Installing/updating required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ An error occurred during installation: {e}")
    except FileNotFoundError:
        print("❌ 'requirements.txt' not found. Please create it first.")

def main_menu():
    """Displays the main menu and handles user input."""
    if not PATHS: 
        print("Error: Path configuration failed.")
        return
    
    set_all_seeds(42)

    login_to_huggingface(PATHS)
    
    while True:
        print("\n==============================")
        print("   OCR Post-Correction Menu")
        print("==============================")
        print("1. Train Model(s)")
        print("2. Evaluate Model(s)")
        print("3. Human vs. Gemini Correlation")
        print("4. Install/Update Requirements")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")

        if choice == "1": handle_train_model()
        elif choice == "2": handle_evaluate_model()
        elif choice == "3": handle_human_correlation()
        elif choice == "4": handle_install_requirements()
        elif choice == "5":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main_menu()