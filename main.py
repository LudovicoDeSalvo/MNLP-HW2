import os
import sys
import subprocess
import pandas as pd
from src.utils import set_all_seeds, login_to_huggingface, configure_gemini, load_config
from src.data_prep import prepare_dataset
from src.train import train_model
from src.eval import evaluate_model, run_human_vs_gemini_correlation, EVAL_RESULTS_DIR

# --- GLOBAL VARIABLES ---
CONFIGS = {
    "1": {"name": "TinyLlama", "path": "config_tinyllama.json"},
    "2": {"name": "Minerva", "path": "config_minerva.json"}
}
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

def select_model_configs():
    """Lets the user select one or more model configurations."""
    print("\nSelect model(s):")
    print("1. TinyLlama")
    print("2. Minerva")
    print("3. Both")
    
    while True:
        choice = input("Enter your choice (1-3): ")
        if choice == "1": return [load_config(CONFIGS["1"]["path"])]
        if choice == "2": return [load_config(CONFIGS["2"]["path"])]
        if choice == "3": return [load_config(CONFIGS["1"]["path"]), load_config(CONFIGS["2"]["path"])]
        print("Invalid choice. Please enter a number between 1 and 3.")

def select_evaluation_file():
    """Lets the user select a single evaluation file to analyze."""
    if not os.path.exists(EVAL_RESULTS_DIR):
        print(f"❌ The '{EVAL_RESULTS_DIR}' directory does not exist. Run an evaluation first.")
        return None, None
    
    eval_files = [f for f in os.listdir(EVAL_RESULTS_DIR) if f.endswith('.csv')]
    if not eval_files:
        print(f"❌ No evaluation files found in the '{EVAL_RESULTS_DIR}' directory.")
        return None, None
    
    print("\nPlease select an evaluation file to analyze:")
    for i, f in enumerate(eval_files):
        print(f"  {i+1}: {f}")
    
    while True:
        try:
            choice = int(input(f"Enter number (1-{len(eval_files)}): ")) - 1
            if 0 <= choice < len(eval_files):
                filename = eval_files[choice]
                # Determine dataset from filename, e.g., 'eval_TinyLlama..._eng.csv'
                dataset_key = 'ita' if '_ita.csv' in filename else 'eng'
                return os.path.join(EVAL_RESULTS_DIR, filename), dataset_key
            print("Invalid number.")
        except (ValueError, IndexError):
            print("Please enter a valid number.")

# --- MENU HANDLERS ---
def handle_train_model():
    """Handler for training models."""
    dataset_key = select_dataset()
    selected_configs = select_model_configs()
    for config in selected_configs:
        if not config: continue
        print(f"\n--- Preparing data for {config['model_name']} on '{dataset_key}' dataset ---")
        train_ds, eval_ds = prepare_dataset(config, dataset_key)
        if train_ds:
            train_model(
                model_name=config['model_name'],
                output_dir=config['output_dir'],
                train_dataset=train_ds,
                eval_dataset=eval_ds
            )

def handle_evaluate_model():
    """Handler for evaluating models."""
    global GEMINI_MODEL
    dataset_key = select_dataset()
    selected_configs = select_model_configs()
    use_gemini = ask_yes_no("⭐ Use Gemini for scoring? (Requires API key)")
    
    if use_gemini and not GEMINI_MODEL:
        GEMINI_MODEL = configure_gemini()
        if not GEMINI_MODEL:
            print("⚠️ Cannot proceed with Gemini scoring. Continuing without it.")
            use_gemini = False

    for config in selected_configs:
        if not config: continue
        print(f"\n--- Preparing data for {config['model_name']} on '{dataset_key}' dataset ---")
        _, eval_ds = prepare_dataset(config, dataset_key)
        if eval_ds:
            evaluate_model(config, dataset_key, eval_ds, GEMINI_MODEL, use_gemini)

def handle_correlation_analysis():
    """Handler for human vs. Gemini correlation."""
    print("\n--- Human vs. Gemini Correlation Analysis ---")
    eval_file_path, dataset_key = select_evaluation_file()
    if not eval_file_path:
        return

    # Load a config to find the human annotations file path
    config = load_config(CONFIGS["1"]["path"]) 
    if not config: return
    
    human_annotations_path = config["datasets"][dataset_key]["human_annotations_path"]
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
    set_all_seeds(42)
    login_to_huggingface()
    
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