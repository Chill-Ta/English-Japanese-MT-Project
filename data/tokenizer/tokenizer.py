import os
import random
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC

def batch_iterator(data_path, max_lines=1_000_000):
    """
    Generator function to read and randomly sample lines from the dataset.
    Uses Reservoir Sampling to prevent Out of Memory (OOM) errors when 
    processing extremely large datasets (e.g., 20M+ lines).
    """
    print(f"Reading and sampling dataset using Reservoir Sampling. Target: {max_lines:,} lines...")
    sampled_lines = []
    
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # Fill the reservoir up to max_lines
            if i < max_lines:
                sampled_lines.append(line)
            else:
                # Randomly replace elements with decreasing probability
                j = random.randint(0, i)
                if j < max_lines:
                    sampled_lines[j] = line

    count = 0
    for line in sampled_lines:
        parts = line.strip('\n').split('\t')
        if len(parts) >= 5:
            yield parts[3]  # English
            yield parts[4]  # Japanese
            count += 1

def train_tokenizer(data_path, save_path, vocab_size=50000, max_lines=1_000_000):
    print(f"Running Tokenizer from {data_path} (Limit {max_lines:,} lines)...")
    
    # 1. Add Special Tokens for bidirectional translation
    special_tokens = ["<pad>", "<s>", "</s>", "<unk>", "<mask>", "<2en>", "<2ja>"]
    
    # 2. Create BPE Tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.normalizer = normalizers.Sequence([NFKC()])
    
    # 3. UnicodeScripts + Metaspace
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.UnicodeScripts(),
        pre_tokenizers.Metaspace(replacement=" ")
    ])
    
    # 4. Configure Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # 5. Train via Generator
    tokenizer.train_from_iterator(batch_iterator(data_path, max_lines), trainer=trainer)

    # Embed the Decoder directly into the Tokenizer
    tokenizer.decoder = decoders.Metaspace(replacement=" ")

    # 6. Save Tokenizer
    tokenizer.save(save_path)
    print(f"Saved Tokenizer at: {save_path}")

if __name__ == "__main__":
    DATA_PATH = "data/dataset/train-set/en-ja/resources/en-ja_extracted_filtered.txt"
    SAVE_PATH = "data/tokenizer/shared_tokenizer.json"
    
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    train_tokenizer(DATA_PATH, SAVE_PATH)