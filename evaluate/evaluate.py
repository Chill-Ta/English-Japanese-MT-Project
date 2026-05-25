import torch
import sacrebleu
import argparse
import re
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from tokenizers import Tokenizer
from model.transformer import Transformer
from comet import download_model, load_from_checkpoint

# ==========================================
# 1. Argument Parser Configuration
# ==========================================
parser = argparse.ArgumentParser(description="Evaluate Transformer MT Model")

parser.add_argument("--direction", type=str, default=None, choices=["en-ja", "ja-en"],
                    help="Direction: en-ja (English->Japanese) or ja-en (Japanese->English)")

parser.add_argument("--src_col", type=int, default=None, help="Source column (0-indexed)")
parser.add_argument("--tgt_col", type=int, default=None, help="Target/Reference column (0-indexed)")
parser.add_argument("--has_header", action="store_true", help="Set this flag if the file contains a header row")

# --- Arguments for Beam Search ---
parser.add_argument("--beam_width", type=int, default=3, help="Beam size for Beam Search (e.g., 3, 5)")
parser.add_argument("--alpha", type=float, default=0.75, help="Length penalty parameter for Beam Search")

args = parser.parse_args()

if args.direction is None:
    print("\n" + "="*40)
    print("Please select translation direction:")
    print("  [1] English - Japanese (en-ja)")
    print("  [2] Japanese - English (ja-en)")
    print("="*40)
    
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == '1':
            args.direction = "en-ja"
            break
        elif choice == '2':
            args.direction = "ja-en"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if args.direction == "en-ja":
    # EN-JA: source = English (Column 0), target = Japanese (Column 1)
    args.src_col = 0 if args.src_col is None else args.src_col
    args.tgt_col = 1 if args.tgt_col is None else args.tgt_col
elif args.direction == "ja-en":
    # JA-EN: source = Japanese (Column 1), target = English (Column 0)
    args.src_col = 1 if args.src_col is None else args.src_col
    args.tgt_col = 0 if args.tgt_col is None else args.tgt_col


# ==========================================
# 2. Configurations
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TOKENIZER_FILE = "data/tokenizer/shared_tokenizer.json"
TEST_FILE = "data/dataset/test_merged.txt"
MODEL = "checkpoints/transformer_epoch_5.pt"


D_MODEL = 256
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 1024
MAX_LENGTH = 128

# ==========================================
# 3. Load Tokenizer and Model
# ==========================================
print(f"Preparing Model for translating: {args.direction}...")
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
vocab_size = tokenizer.get_vocab_size()
pad_idx = tokenizer.token_to_id("<pad>")

model = Transformer(
    src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
    src_pad_idx=pad_idx, tgt_pad_idx=pad_idx,
    d_model=D_MODEL, num_layers=NUM_LAYERS, 
    num_heads=NUM_HEADS, d_ff=D_FF
).to(DEVICE)

# Load Checkpoint
checkpoint = torch.load(MODEL, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==========================================
# 4. Translation Function
# ==========================================
def translate(sentence, target_lang="ja", beam_width=3, max_length=MAX_LENGTH, alpha=0.75):
    tag = "<2ja>" if target_lang == "ja" else "<2en>"
    tag_id = tokenizer.token_to_id(tag)
    
    src_tokens = tokenizer.encode(sentence).ids
    src_ids = [tokenizer.token_to_id("<s>"), tag_id] + src_tokens + [tokenizer.token_to_id("</s>")]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    src_mask = model.make_src_mask(src_tensor)
    with torch.no_grad():
        enc_output = model.encoder(src_tensor, src_mask)
        
    start_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    
    # Beams are stored as a list of Tuples: (sequence_list, log_score)
    beams = [([start_id], 0.0)]
    completed_beams = []
    
    for step in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            # If this beam is finished (EOS reached), save to completed_beams and skip
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
                continue
                
            tgt_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            with torch.no_grad():
                output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
                
                # Extract logits for the latest token
                logits = output[0, -1, :] 
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Find the Top-K most probable next tokens for this sequence
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for prob, token_id in zip(topk_probs, topk_ids):
                    new_seq = seq + [token_id.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
                    
        # Stop if no further beams can be generated
        if len(new_beams) == 0:
            break
            
        # Sort scores (descending) and keep only the top 'beam_width' number of beams
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        
    # Fallback in case MAX_LENGTH is reached without finding </s>
    if len(completed_beams) == 0:
        completed_beams = beams
        
    # Final score calculation function (prevents bias towards short sentences)
    def calc_final_score(seq, score):
        length = len(seq) - 1 # Exclude <s>
        if length == 0: return -1e9
        return score / (length ** alpha)
        
    # Select the beam with the best score
    best_beam = max(completed_beams, key=lambda x: calc_final_score(x[0], x[1]))
    best_seq = best_beam[0]
    
    # Decode back to text (The new Tokenizer Decoder will automatically reconstruct words!)
    return tokenizer.decode(best_seq[1:-1], skip_special_tokens=True)

# ==========================================
# 5. Start Evaluation
# ==========================================
if __name__ == "__main__":
    references = []
    predictions = []
    sources = []
    
    sample_outputs = []

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Evaluation with test file {TEST_FILE}...")
    print(f"Using Beam Search (Width={args.beam_width}, Alpha={args.alpha})")
    
    for i, line in enumerate(tqdm(lines)):
        if i == 0 and args.has_header: 
            continue
        
        parts = line.strip('\n').split('\t')
        max_col_needed = max(args.src_col, args.tgt_col)

        if len(parts) > max_col_needed:
            source_text = parts[args.src_col] 
            truth_text = parts[args.tgt_col]
        
            target_lang = "ja" if args.direction == "en-ja" else "en"

            # Execute Beam Search
            pred_text = translate(
                source_text, 
                target_lang=target_lang, 
                beam_width=args.beam_width, 
                alpha=args.alpha
            )
            
            # ==========================================
            # Post-processing (Simplified for Metaspace Tokenizer)
            # ==========================================
            clean_pred = pred_text.strip()
            
            if target_lang == "ja":
                # Japanese doesn't use spaces, remove them safely
                clean_pred = clean_pred.replace(" ", "")
                truth_text = truth_text.replace(" ", "") 
                # Remove accidental English characters
                clean_pred = re.sub(r'[a-zA-Z]+', '', clean_pred)

            if target_lang == "en":
                # 1. Clean up BPE artifacts if they somehow leak through
                clean_pred = clean_pred.replace("@@ ", "")
                
                # NOTE: No more manual space removal needed! 
                # Words like "acknow led ged" are already fixed by the tokenizer.
                
                # 2. Fix spacing around punctuation marks
                clean_pred = re.sub(r'\s+([.,?!])', r'\1', clean_pred)
                truth_text = re.sub(r'\s+([.,?!])', r'\1', truth_text)
                
                # 3. Fix loose apostrophes (e.g., didn 't -> didn't)
                clean_pred = clean_pred.replace(" '", "'").replace("' ", "'").strip()
                
                # 4. Clear redundant spaces
                clean_pred = re.sub(r'\s+', ' ', clean_pred).strip()
            
            predictions.append(clean_pred)
            references.append(truth_text)
            sources.append(source_text)

            if len(predictions) <= 5:
                sample_outputs.append({
                    "id": len(predictions),
                    "source": source_text,
                    "pred": clean_pred,
                    "ref": truth_text
                })

# ==========================================
# 6. Display All Results
# ==========================================
    print("\n" + "="*50)
    print(f" Translation Examples (direction: {args.direction})")
    print("="*50)
    
    for sample in sample_outputs:
        print(f"[Sample no. {sample['id']}]")
        print(f"S (Source) : {sample['source']}")
        print(f"P (Pred)   : {sample['pred']}")
        print(f"R (Ref)    : {sample['ref']}")
        print("-" * 30)

    print("\n" + "="*50)
    print(f"Evaluation Metrics")
    print("="*50)

    if args.direction == "en-ja":
        bleu_tok = 'intl' 
    else:
        bleu_tok = '13a'
    
    try:
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize=bleu_tok)
        print(f"BLEU Score  (tok:{bleu_tok}): {bleu.score:.2f}")
    except Exception as e:
        print(f"BLEU calculation failed with {bleu_tok}: {e}")
        print("Try using 'intl' tokenizer instead...")
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='intl')
        print(f"BLEU Score  (tokenizer:intl): {bleu.score:.2f}")
    
    chrf2 = sacrebleu.corpus_chrf(predictions, [references])
    print(f"chrF2 Score : {chrf2.score:.2f}")

    ter = sacrebleu.corpus_ter(predictions, [references])
    print(f"TER Score   : {ter.score:.2f}")
    print("="*50)

    # ==========================================
    # 7. COMET Score 
    # ==========================================
    print("\n" + "="*50)
    print("Calculating COMET Score...")
    print("="*50)
    
    try:
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)
        
        comet_data = []
        for src, mt, ref in zip(sources, predictions, references):
            comet_data.append({"src": src, "mt": mt, "ref": ref})
            
        comet_output = comet_model.predict(comet_data, batch_size=8, gpus=1)
        
        print(f"COMET Score : {comet_output.system_score:.4f}")
        
    except Exception as e:
        print(f"COMET calculation failed: {e}")
        print("Help: If Error 'CUDA Out of Memory' appear, Please change gpus=1 to gpus=0")
    
    print("="*50)