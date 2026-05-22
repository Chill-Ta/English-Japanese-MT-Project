import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import random
from itertools import islice
import torch.multiprocessing as mp
import platform

# Fix: Must be a top-level function only
# Windows uses 'spawn' instead of 'fork', making it impossible to pickle a lambda inside a function
def _worker_init_fn(worker_id):
    random.seed(torch.initial_seed() % 2**32)

class TranslationIterableDataset(IterableDataset):
    def __init__(self, filepath, src_tokenizer_path, tgt_tokenizer_path, max_length=128, skip_lines=0, min_score=0.70):
        self.filepath = filepath
        self.max_length = max_length
        self.skip_lines = skip_lines
        self.min_score = min_score
        
        self.src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
        self.tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
        
        self.src_bos = self.src_tokenizer.token_to_id("<s>")
        self.src_eos = self.src_tokenizer.token_to_id("</s>")
        self.tgt_bos = self.tgt_tokenizer.token_to_id("<s>")
        self.tgt_eos = self.tgt_tokenizer.token_to_id("</s>")

        # Verify special tokens immediately during initialization
        # Prevents tag_id from being None and causing torch.tensor() to crash without a clear error message
        for token in ["<s>", "</s>", "<pad>", "<2en>", "<2ja>"]:
            assert self.src_tokenizer.token_to_id(token) is not None, \
                f"Special token '{token}' not found in tokenizer! please retrain tokenizer"

    def __iter__(self):
        with open(self.filepath, 'r', encoding='utf-8') as f:
            # Use islice instead of a next() loop — many times faster on large datasets
            for line in islice(f, self.skip_lines, None):
                parts = line.strip('\n').split('\t')
                
                if len(parts) < 5:
                    continue 
                
                # Filter out low-quality sentences (Quality Filter)
                try:
                    score = float(parts[2])
                    if score < self.min_score:
                        continue 
                except ValueError:
                    continue
                
                eng_text = parts[3] 
                jap_text = parts[4]

                # 50/50 random split to toggle translation direction
                if random.random() < 0.5:
                    source_sentence = eng_text
                    target_sentence = jap_text
                    tag_id = self.src_tokenizer.token_to_id("<2ja>")
                else:
                    source_sentence = jap_text
                    target_sentence = eng_text
                    tag_id = self.src_tokenizer.token_to_id("<2en>")

                src_encoded = self.src_tokenizer.encode(source_sentence).ids[:self.max_length-3]
                tgt_encoded = self.tgt_tokenizer.encode(target_sentence).ids[:self.max_length-2]
                
                src_ids = [self.src_bos, tag_id] + src_encoded + [self.src_eos]
                tgt_ids = [self.tgt_bos] + tgt_encoded + [self.tgt_eos]
                
                yield torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

class CollateFunc:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch, tgt_batch = [], []
        for src_item, tgt_item in batch:
            src_batch.append(src_item)
            tgt_batch.append(tgt_item)
            
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx)
        
        return src_padded, tgt_padded

def get_dataloader(filepath, src_tok_path, tgt_tok_path, batch_size=32, skip_lines=0, min_score=0.65):
    dataset = TranslationIterableDataset(filepath, src_tok_path, tgt_tok_path, skip_lines=skip_lines, min_score=min_score)
    
    # Reuse the tokenizer already loaded in the dataset instead of reloading it
    pad_idx = dataset.src_tokenizer.token_to_id("<pad>")
    
    # Fix: Windows uses num_workers=0 due to multiprocessing behavior on Windows
    # Using 'spawn' instead of 'fork' prevents pickling and introduces high overhead.
    # Linux/Mac use 'fork', so they can use multi-worker processes normally.
    is_windows = platform.system() == "Windows"
    num_workers = 0 if is_windows else min(4, mp.cpu_count())
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=CollateFunc(pad_idx),
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None
    )
    return dataloader, pad_idx