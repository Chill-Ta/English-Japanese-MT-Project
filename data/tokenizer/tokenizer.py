from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
# Removed StripAccents — it destroys Japanese characters (e.g., が→か, ば→は)
from tokenizers.normalizers import NFKC
import os

def batch_iterator(data_path, max_lines=1_000_000):
    """
    Generator function to read file line by line.
    Extracts only EN column (index 3) and JA column (index 4).
    Prevents metadata/score noise from being fed into the tokenizer learning process.
    """
    count = 0
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if count >= max_lines:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 5:
                yield parts[3]  # English
                yield parts[4]  # Japanese
                count += 1

def train_tokenizer(data_path, save_path, vocab_size=32000, max_lines=1_000_000):
    print(f"Running Tokenizer from {data_path} (Limit {max_lines:,} lines)...")
    
    # 1. Add Special Tokens for bidirectional translation
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<2en>", "<2ja>"]
    
    # 2. Create BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    # Use NFKC normalization only — safe for both Japanese and English
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    
    # Fix: UnicodeScripts + Metaspace is best for ja-en
    # Removed 'add_prefix_space=True' as Metaspace handles it by default
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Metaspace(replacement=" ")
    ])
    
    # 3. Configure Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # 4. Train via Generator
    tokenizer.train_from_iterator(batch_iterator(data_path, max_lines), trainer=trainer)

    # Embed the Decoder directly into the Tokenizer.
    # Removed 'add_prefix_space=True' here as well
    tokenizer.decoder = decoders.Metaspace(replacement=" ")

    # 5. Save Tokenizer
    tokenizer.save(save_path)
    print(f"Saved Tokenizer at: {save_path}")

if __name__ == "__main__":
    source_file = "data/dataset/en-ja.bicleaner05.txt"
    output_file = "data/shared_tokenizer.json"
    
    train_tokenizer(
        data_path=source_file, 
        save_path=output_file, 
        vocab_size=32000, 
        max_lines=1_000_000
    )