import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import random
import platform

def _worker_init_fn(worker_id):
    random.seed(torch.initial_seed() % 2**32)

class TranslationIterableDataset(IterableDataset):
    def __init__(self, filepath, src_tokenizer_path, tgt_tokenizer_path, max_length=128, skip_lines=0, min_score=0.70, buffer_size=2048):
        self.filepath = filepath
        self.max_length = max_length
        self.skip_lines = skip_lines
        self.min_score = min_score
        self.buffer_size = buffer_size
        
        self.src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
        self.tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
        
        self.src_bos = self.src_tokenizer.token_to_id("<s>")
        self.src_eos = self.src_tokenizer.token_to_id("</s>")
        self.tgt_bos = self.tgt_tokenizer.token_to_id("<s>")
        self.tgt_eos = self.tgt_tokenizer.token_to_id("</s>")

    def process_line(self, line):
        parts = line.strip('\n').split('\t')
        if len(parts) >= 5:
            try:
                score = float(parts[2])
                if score < self.min_score:
                    return None
            except ValueError:
                return None
                
            src_text = parts[3]
            tgt_text = parts[4]
            
            src_enc = self.src_tokenizer.encode(src_text).ids
            tgt_enc = self.tgt_tokenizer.encode(tgt_text).ids
            
            if len(src_enc) > self.max_length - 2 or len(tgt_enc) > self.max_length - 2:
                return None
                
            src_tensor = torch.tensor([self.src_bos] + src_enc + [self.src_eos], dtype=torch.long)
            tgt_tensor = torch.tensor([self.tgt_bos] + tgt_enc + [self.tgt_eos], dtype=torch.long)
            return src_tensor, tgt_tensor
        return None

    def __iter__(self):
        buffer = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for _ in range(self.skip_lines):
                next(f, None)
                
            for line in f:
                processed = self.process_line(line)
                if processed is not None:
                    buffer.append(processed)
                    
                # Implement Bucket Batching: Sort buffer by source length to minimize padding
                if len(buffer) >= self.buffer_size:
                    buffer.sort(key=lambda x: len(x[0]))
                    for item in buffer:
                        yield item
                    buffer = []
            
            # Yield remaining items in the buffer
            if buffer:
                buffer.sort(key=lambda x: len(x[0]))
                for item in buffer:
                    yield item

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_padded = pad_sequence(src_batch, batch_first=True, padding_value=self.pad_idx)
        tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=self.pad_idx)
        return src_padded, tgt_padded

def get_dataloader(filepath, src_tok_path, tgt_tok_path, batch_size=32, skip_lines=0, min_score=0.70):
    dataset = TranslationIterableDataset(filepath, src_tok_path, tgt_tok_path, skip_lines=skip_lines, min_score=min_score)
    pad_idx = dataset.src_tokenizer.token_to_id("<pad>")
    
    is_windows = platform.system() == "Windows"
    num_workers = 0 if is_windows else min(4, torch.multiprocessing.cpu_count())
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=PadCollate(pad_idx),
        num_workers=num_workers,
        worker_init_fn=_worker_init_fn
    )
    return dataloader, pad_idx