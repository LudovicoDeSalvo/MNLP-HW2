import json
import os

# Load JSON data from the correct paths
ocr_path = os.path.join("dataset", "ita", "original_ocr.json")
cleaned_path = os.path.join("dataset", "ita", "cleaned.json")

with open(ocr_path, "r", encoding="utf-8") as f:
    ocr_data = json.load(f)

with open(cleaned_path, "r", encoding="utf-8") as f:
    cleaned_data = json.load(f)

# Collect common keys between the two datasets
common_keys = sorted(set(ocr_data) & set(cleaned_data))

# Pair full paragraphs directly (no sentence splitting, full alignment)
paragraph_pairs = []
for key in common_keys:
    ocr_paragraph = ocr_data[key].strip()
    cleaned_paragraph = cleaned_data[key].strip()
    paragraph_pairs.append({
        "id": key,
        "ocr": ocr_paragraph,
        "cleaned": cleaned_paragraph
    })

# Manual annotation for all paragraph pairs
annotations = []
print("\nRate each OCR-cleaned paragraph pair with a score from 0 to 5.\n")
for entry in paragraph_pairs:
    print("OCR Paragraph:\n", entry["ocr"])
    print("Cleaned Paragraph:\n", entry["cleaned"])
    while True:
        try:
            score = int(input("Your Score [0–5]: "))
            if 0 <= score <= 5:
                break
            print("Score must be between 0 and 5.")
        except ValueError:
            print("Invalid input. Enter an integer between 0 and 5.")
    annotations.append({**entry, "human_score": score})

# Save the annotated output to the target directory
os.makedirs("general_utils", exist_ok=True)
output_path = os.path.join("general_utils", "human_annotations.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(annotations, f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(annotations)} human-annotated paragraph pairs to {output_path}")
