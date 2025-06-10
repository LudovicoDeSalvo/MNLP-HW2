import torch
from wtpsplit import SaT as SaTSplitter
from sentence_transformers import SentenceTransformer, util


def load_sat_splitter(model_name="sat-3l"):
    print(f"🔍 Loading SaT model ({model_name})...")
    
    sat = SaTSplitter(model_name)          # still the wrapper
    if torch.cuda.is_available():
        sat.model.half().cuda()            # move ONLY the inner HF model
    return sat                             # wrapper is intact

def split_with_sat(text, tokenizer, model):
    tokens = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)[0].tolist()
    offsets = tokens["offset_mapping"][0].tolist()
    input_ids = tokens["input_ids"][0].tolist()

    sentences = []
    current_sentence = ""
    for idx, label in enumerate(preds):
        start, end = offsets[idx]
        if start == end:
            continue
        token_text = tokenizer.convert_ids_to_tokens([input_ids[idx]])[0].replace("▁", " ").strip()
        current_sentence += token_text + " "
        if label == 0:  # B-SENT
            if current_sentence.strip():
                sentences.append(current_sentence.strip())
            current_sentence = ""
    if current_sentence.strip():
        sentences.append(current_sentence.strip())
    return sentences

def align_sentences_by_semantics(noisy_sents, clean_sents, model, threshold=0.5):
    """
    Aligns noisy sentences to clean ones using cosine similarity of embeddings.
    Returns a list of (noisy, clean) pairs.
    """
    noisy_embeddings = model.encode(noisy_sents, convert_to_tensor=True)
    clean_embeddings = model.encode(clean_sents, convert_to_tensor=True)

    cosine_scores = util.cos_sim(noisy_embeddings, clean_embeddings)

    aligned_pairs = []
    used_clean = set()

    for i, scores in enumerate(cosine_scores):
        best_match_id = int(torch.argmax(scores))
        best_score = float(scores[best_match_id])

        if best_score > threshold and best_match_id not in used_clean:
            aligned_pairs.append((noisy_sents[i], clean_sents[best_match_id]))
            used_clean.add(best_match_id)

    return aligned_pairs
