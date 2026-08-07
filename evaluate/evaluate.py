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
# MeCab Tokenizer for Japanese BLEU
# ==========================================
# Standard SacreBLEU evaluation for Japanese requires morphological
# segmentation before BLEU computation. Without this, BLEU treats
# Japanese as character-level, destroying n-gram overlap and producing
# artificially low scores (e.g., 1.71 instead of ~20+).
try:
    import fugashi
    _MECAB_TAGGER = fugashi.Tagger()
    _MECAB_AVAILABLE = True
except ImportError:
    _MECAB_TAGGER = None
    _MECAB_AVAILABLE = False
    print("WARNING: fugashi (MeCab) not installed. Japanese BLEU will use fallback tokenization.")
    print("         Install with: pip install fugashi unidic-lite")

def tokenize_japanese(text: str) -> str:
    """
    Tokenize Japanese text into space-separated morphemes using MeCab.
    This is required for meaningful BLEU calculation on Japanese text.
    Falls back to character-level tokenization if MeCab is unavailable.
    """
    if _MECAB_AVAILABLE and _MECAB_TAGGER is not None:
        # MeCab returns Word objects; extract surface forms
        morphemes = [word.surface for word in _MECAB_TAGGER(text) if word.surface.strip()]
        return " ".join(morphemes)
    else:
        # Fallback: character-level (better than nothing, but still suboptimal)
        return " ".join(list(text.replace(" ", "")))

# ==========================================
# 1. Argument Parser Configuration
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Transformer MT Model")

    parser.add_argument("--direction", type=str, default=None, choices=["en-ja", "ja-en"],
                        help="Direction: en-ja (English->Japanese) or ja-en (Japanese->English)")
    parser.add_argument("--src_col", type=int, default=None, help="Source column (0-indexed)")
    parser.add_argument("--tgt_col", type=int, default=None, help="Target/Reference column (0-indexed)")
    parser.add_argument("--has_header", action="store_true", help="Set this flag if the file contains a header row")

    parser.add_argument("--beam_width", type=int, default=3, help="Beam size for Beam Search")
    parser.add_argument("--alpha", type=float, default=0.75, help="Length penalty parameter")
    parser.add_argument("--rep_penalty", type=float, default=1.3, help="Repetition penalty ( > 1.0 to penalize)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for softmax scaling (< 1.0 = sharper, > 1.0 = smoother)")

    parser.add_argument("--style", type=str, default="plain", choices=["plain", "polite"],
                        help="Japanese style for EN->JA: plain (だ/た) or polite (です/ます). Default: plain")

    parser.add_argument("--export", action="store_true", help="Export source/prediction/reference to TSV for MQM analysis")
    parser.add_argument("--export_path", type=str, default="evaluate/mt_output_for_mqm.tsv", help="Path for export TSV file")

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
        args.src_col = 0 if args.src_col is None else args.src_col
        args.tgt_col = 1 if args.tgt_col is None else args.tgt_col
    elif args.direction == "ja-en":
        args.src_col = 1 if args.src_col is None else args.src_col
        args.tgt_col = 0 if args.tgt_col is None else args.tgt_col

    return args

# ==========================================
# 2. Configurations (Global variables)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Prefer the v3 tokenizer (with style tags) if available, else fall back to v2
_TOK_V3 = "data/tokenizer/shared_tokenizerv3.json"
_TOK_V2 = "data/tokenizer/shared_tokenizerv2.json"
TOKENIZER_FILE = _TOK_V3 if os.path.exists(_TOK_V3) else _TOK_V2
TEST_FILE = "data/dataset/test-set/test_merged.txt"
MODEL_PATH = "checkpoints/transformer_epoch_10_chunk_16.pt" # Update to your chunk checkpoint name

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
MAX_LENGTH = 128

# Must match train.py — set True if the checkpoint was trained with RoPE
USE_ROPE = False

# ==========================================
# 2.5 Post-Processing: EN Token Leaking Cleanup
# ==========================================
# Common English words that leak into Japanese output untranslated.
# Maps isolated English tokens to their most common Japanese equivalents.
_EN_LEAK_MAP = {
    "true": "真実", "false": "偽", "good": "良い", "bad": "悪い",
    "yes": "はい", "no": "いいえ", "ok": "了解", "nice": "素敵",
    "the": "", "a": "", "an": "", "is": "", "are": "", "was": "",
    "and": "", "or": "", "but": "", "not": "",
}

# Regex: detect isolated ASCII words surrounded by Japanese characters
_JA_SCRIPT_RE = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF。、]')
_ISOLATED_EN_RE = re.compile(r'(?<=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF。、])\s*([A-Za-z]+)\s*(?=[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF。、])')

def clean_ja_english_leaking(text: str) -> str:
    """
    Remove isolated English words that leaked into Japanese output.
    Only removes words that are surrounded by Japanese characters on both sides.
    Preserves legitimate loanwords written in ASCII (proper nouns, brands).
    """
    def _replace(m):
        word = m.group(1).lower()
        if word in _EN_LEAK_MAP:
            return _EN_LEAK_MAP[word]
        # If the word is very short (1-3 chars) and not a known abbreviation, remove it
        if len(word) <= 3 and not word.isupper():
            return ""
        # Keep longer words and acronyms (could be proper nouns)
        return m.group(0)
    
    return _ISOLATED_EN_RE.sub(_replace, text)

# ==========================================
# 3. Translation Function
# ==========================================
def translate(sentence, model, tokenizer, target_lang="ja", beam_width=3, max_length=MAX_LENGTH, alpha=0.75, rep_penalty=1.5, temperature=1.0, style="plain"):
    # Style-aware tag selection for EN->JA direction
    if target_lang == "ja":
        style_tag = f"<2ja_{style}>"
        tag_id = tokenizer.token_to_id(style_tag)
        # Fallback: if the style-specific tag doesn't exist (old tokenizer), use <2ja>
        if tag_id is None:
            tag_id = tokenizer.token_to_id("<2ja>")
    else:
        tag_id = tokenizer.token_to_id("<2en>")
    
    src_tokens = tokenizer.encode(sentence).ids
    src_ids = [tokenizer.token_to_id("<s>"), tag_id] + src_tokens + [tokenizer.token_to_id("</s>")]
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.inference_mode():
        enc_output = model.encoder(src_tensor, src_mask)
        
    start_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    pad_idx = tokenizer.token_to_id("<pad>")
    
    beams = [([start_id], 0.0)]
    completed_beams = []
    
    for step in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq[-1] == eos_id:
                completed_beams.append((seq, score))
                continue
                
            tgt_tensor = torch.tensor(seq, dtype=torch.long, device=DEVICE).unsqueeze(0)
            tgt_mask = model.make_tgt_mask(tgt_tensor)
            
            with torch.inference_mode():
                output = model.decoder(tgt_tensor, enc_output, src_mask, tgt_mask)
                logits = output[0, -1, :] 
                
                # Repetition Penalty logic with exclusion for special tokens
                if rep_penalty != 1.0:
                    special_tokens = {start_id, eos_id, pad_idx, tag_id}
                    for token_id in set(seq):
                        if token_id not in special_tokens:
                            if logits[token_id] > 0:
                                logits[token_id] /= rep_penalty
                            else:
                                logits[token_id] *= rep_penalty
                
                # Temperature scaling for output diversity control
                if temperature != 1.0:
                    logits = logits / temperature
                
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, beam_width)
                
                for prob, token_id in zip(topk_probs, topk_ids):
                    new_seq = seq + [token_id.item()]
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
                    
        if len(new_beams) == 0:
            break
            
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        
    if len(completed_beams) == 0:
        completed_beams = beams
        
    def calc_final_score(seq, score):
        length = max(1, len(seq) - 1) 
        return score / (length ** alpha)
        
    best_beam = max(completed_beams, key=lambda x: calc_final_score(x[0], x[1]))
    best_seq = best_beam[0]
    
    return tokenizer.decode(best_seq[1:-1], skip_special_tokens=True)

# ==========================================
# 4. Main Evaluation Execution
# ==========================================
def main():
    args = parse_args()
    
    print(f"Preparing Model for translating: {args.direction}...")
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("<pad>")

    model = Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        src_pad_idx=pad_idx, tgt_pad_idx=pad_idx,
        d_model=D_MODEL, num_layers=NUM_LAYERS, 
        num_heads=NUM_HEADS, d_ff=D_FF,
        use_rope=USE_ROPE
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    references, predictions, sources, sample_outputs = [], [], [], []

    with open(TEST_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Evaluation with test file {TEST_FILE}...")
    print(f"Using Beam Search (Width={args.beam_width}, Alpha={args.alpha}, RepPenalty={args.rep_penalty})")
    
    for i, line in enumerate(tqdm(lines)):
        if i == 0 and args.has_header: 
            continue
        
        parts = line.strip('\n').split('\t')
        max_col_needed = max(args.src_col, args.tgt_col)

        if len(parts) > max_col_needed:
            source_text = parts[args.src_col] 
            truth_text = parts[args.tgt_col]
            target_lang = "ja" if args.direction == "en-ja" else "en"

            pred_text = translate(
                sentence=source_text, model=model, tokenizer=tokenizer,
                target_lang=target_lang, beam_width=args.beam_width, 
                alpha=args.alpha, rep_penalty=args.rep_penalty,
                temperature=args.temperature,
                style=args.style
            )
            
            # --- Post-processing ---
            clean_pred = pred_text.strip()
            
            if target_lang == "ja":
                clean_pred = clean_pred.replace(" .", "。").replace(".", "。").replace(" ", "")
                
                # Cleanup duplicated Japanese punctuations
                clean_pred = re.sub(r'。+', '。', clean_pred)
                clean_pred = re.sub(r'、+', '、', clean_pred)

                # Remove leaked English tokens surrounded by Japanese characters
                clean_pred = clean_ja_english_leaking(clean_pred)

                truth_text = truth_text.replace(" ", "") 
                # Note: English characters are intentionally preserved for valid
                # loanwords, brand names, and proper nouns in Japanese text

            if target_lang == "en":
                clean_pred = clean_pred.replace("@@ ", "")
                clean_pred = re.sub(r'\s+([.,?!])', r'\1', clean_pred)
                truth_text = re.sub(r'\s+([.,?!])', r'\1', truth_text)
                clean_pred = clean_pred.replace(" '", "'").replace("' ", "'").strip()
                clean_pred = re.sub(r'\s+', ' ', clean_pred).strip()
            
            predictions.append(clean_pred)
            references.append(truth_text)
            sources.append(source_text)

            if len(predictions) <= 5:
                sample_outputs.append({
                    "id": len(predictions),
                    "source": source_text, "pred": clean_pred, "ref": truth_text
                })

    # --- Display & Metrics ---
    print("\n" + "="*50)
    print(f" Translation Examples (direction: {args.direction})")
    print("="*50)
    for sample in sample_outputs:
        print(f"[Sample no. {sample['id']}]\nS: {sample['source']}\nP: {sample['pred']}\nR: {sample['ref']}\n" + "-"*30)

    print("\n" + "="*50 + "\nEvaluation Metrics\n" + "="*50)

    # ── Japanese BLEU: MeCab pre-tokenization ──
    # Standard protocol: tokenize Japanese with MeCab, then compute BLEU
    # with tokenize='none' (already tokenized). This fixes the artificially
    # low BLEU caused by SacreBLEU's 'intl' tokenizer treating Japanese
    # as character-level.
    if args.direction == "en-ja":
        mecab_preds = [tokenize_japanese(p) for p in predictions]
        mecab_refs  = [tokenize_japanese(r) for r in references]
        bleu = sacrebleu.corpus_bleu(mecab_preds, [mecab_refs], tokenize='none')
        bleu_label = "ja-mecab"
        print(f"BLEU Score  (tok:{bleu_label}): {bleu.score:.2f}")
        # Also show intl for backward comparison
        bleu_intl = sacrebleu.corpus_bleu(predictions, [references], tokenize='intl')
        print(f"BLEU Score  (tok:intl, legacy): {bleu_intl.score:.2f}")
    else:
        bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize='13a')
        print(f"BLEU Score  (tok:13a): {bleu.score:.2f}")
    
    print(f"chrF2 Score : {sacrebleu.corpus_chrf(predictions, [references]).score:.2f}")
    print(f"chrF++ Score: {sacrebleu.corpus_chrf(predictions, [references], word_order=2).score:.2f}")
    print(f"TER Score   : {sacrebleu.corpus_ter(predictions, [references]).score:.2f}")
    
    # --- COMET ---
    print("\n" + "="*50 + "\nCalculating COMET Score...\n" + "="*50)
    try:
        comet_model_path = download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(comet_model_path)
        
        comet_data = [{"src": s, "mt": p, "ref": r} for s, p, r in zip(sources, predictions, references)]
        
        num_gpus = 1 if torch.cuda.is_available() else 0
        comet_output = comet_model.predict(comet_data, batch_size=8, gpus=num_gpus)
        print(f"COMET Score : {comet_output.system_score:.4f}")
    except Exception as e:
        print(f"COMET calculation failed: {e}")
    print("="*50)

    # --- Export for MQM Analysis ---
    if args.export:
        export_path = args.export_path
        os.makedirs(os.path.dirname(export_path) if os.path.dirname(export_path) else '.', exist_ok=True)
        with open(export_path, 'w', encoding='utf-8') as ef:
            ef.write("id\tsource\tprediction\treference\n")
            for idx, (s, p, r) in enumerate(zip(sources, predictions, references), 1):
                ef.write(f"{idx}\t{s}\t{p}\t{r}\n")
        print(f"\nExported {len(predictions)} translations to: {export_path}")
        print("Run MQM evaluation with: python evaluate/mqm_evaluate.py --input " + export_path)

if __name__ == "__main__":
    main()