import os
import random
from tokenizers import Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC

def batch_iterator(data_path, max_lines=1_000_000):
    """
    Generator function to read and randomly sample lines from the dataset.
    This ensures the tokenizer learns from the entire 21M dataset distribution
    rather than just the first chunk.
    """
    print("Reading dataset for tokenizer sampling. This might take a moment...")
    with open(data_path, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    
    # Randomly sample lines to ensure diverse vocabulary coverage
    sampled_lines = random.sample(all_lines, min(max_lines, len(all_lines)))
    
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
    # Point this to your full 21M dataset file (before chunking)
    source_file = "data/dataset/train-set/en-ja/resources/en-ja.bicleaner05.txt"
    output_file = "data/tokenizer/shared_tokenizer.json"
    
    train_tokenizer(
        data_path=source_file, 
        save_path=output_file, 
        vocab_size=50000, 
        max_lines=1_000_000
    )