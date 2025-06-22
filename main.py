import os
import sys
import subprocess
import pandas as pd
from datasets import Dataset

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
from eval import evaluate_model, run_human_vs_gemini_correlation


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

def handle_correlation_analysis():
    # FIXED: select_evaluation_file no longer needs an argument
    eval_file_path, dataset_key = select_evaluation_file() 
    if not eval_file_path: return

    # We need a config to find the human annotations file path. Dynamically load the first available one.
    if not MODEL_CONFIGS:
        print("❌ No model configs found, cannot determine human annotations path.")
        return
        
    master_config_path = list(MODEL_CONFIGS.values())[0]['path']
    master_config = load_config(master_config_path)
    if not master_config: return
    
    human_annotations_path = os.path.join(
        PATHS['general_utils_dir'],
        master_config["datasets"][dataset_key]["human_annotations_filename"]
    )
    run_human_vs_gemini_correlation(eval_file_path, human_annotations_path)

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
        elif choice == "3": handle_correlation_analysis()
        elif choice == "4": handle_install_requirements()
        elif choice == "5":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main_menu()