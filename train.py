import torch
import torch.nn as nn
import math
import glob
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import os
import gc
import json
from torch.optim.lr_scheduler import LambdaLR

from model.dataloader import get_dataloader
from model.transformer import Transformer

# ==========================================
# Hyperparameters Configuration
# ==========================================
BATCH_SIZE = 32
ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
MAX_LENGTH = 128

GLOBAL_EPOCHS = 10 

D_MODEL = 512
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 2048
VOCAB_SIZE = 50002  # Extended from 50000 to include <2ja_polite> and <2ja_plain>

CHUNK_DIR = "data/dataset/chunks"

# Prefer v3 tokenizer (with style tags) if available, else fall back to v2
_TOK_V3 = "data/tokenizer/shared_tokenizerv3.json"
_TOK_V2 = "data/tokenizer/shared_tokenizerv2.json"
_ACTIVE_TOK = _TOK_V3 if os.path.exists(_TOK_V3) else _TOK_V2
SRC_TOKENIZER = _ACTIVE_TOK
TGT_TOKENIZER = _ACTIVE_TOK

RESUME_TRAINING = True
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True

# ==========================================
# Helper Function: Fast Line Counter
# ==========================================
def count_lines(filepath):
    """Quickly count the total number of lines in a file to calculate total batches."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def train():
    print(f"Preparing for Training on device: {DEVICE}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

# ==========================================
# 1. Prepare Chunk Files
# ==========================================
    chunk_files = sorted(glob.glob(os.path.join(CHUNK_DIR, "*.txt")))
    if not chunk_files:
        print(f"Error: No chunk files found in {CHUNK_DIR}")
        return
        
    TOTAL_CHUNKS = len(chunk_files)
    print(f"Found {TOTAL_CHUNKS} chunk files to process.")

# ==========================================
# 2. Model, Loss, Optimizer Initialization
# ==========================================
    from tokenizers import Tokenizer
    temp_tok = Tokenizer.from_file(SRC_TOKENIZER)
    pad_idx = temp_tok.token_to_id("<pad>")

    print("Creating Transformer Model...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        src_pad_idx=pad_idx, tgt_pad_idx=pad_idx,
        d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.1
    ).to(DEVICE)
    # Added label_smoothing to prevent the model from becoming overconfident,
    # which helps mitigate literal translation and idiomatic blindness.
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    TOTAL_STEPS = TOTAL_CHUNKS * GLOBAL_EPOCHS
    WARMUP_CHUNKS = max(1, TOTAL_STEPS // 10) 

    def lr_lambda(current_step):
        if current_step < WARMUP_CHUNKS:
            return float(current_step + 1) / float(max(1, WARMUP_CHUNKS))
        progress = float(current_step - WARMUP_CHUNKS) / float(max(1, TOTAL_STEPS - WARMUP_CHUNKS))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler(device='cuda')

# ==========================================
# 3. Checkpoint Load (Resume Training)
# ==========================================
    start_global_epoch = 0
    start_chunk_idx = 0
    global_step = 0
    start_batch_idx = -1 

    if RESUME_TRAINING:
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "transformer_epoch_*_chunk_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
            print(f"Loading Checkpoint From: {latest_checkpoint}")
            
            checkpoint = torch.load(latest_checkpoint, map_location=DEVICE, weights_only=True)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            # ── Embedding Resize (vocab migration) ──
            # If the checkpoint was saved with a smaller vocab (e.g., 50000) but
            # current VOCAB_SIZE is larger (e.g., 50002 for style tags), auto-resize.
            ckpt_vocab = checkpoint.get('vocab_size_migrated_to', None)
            if ckpt_vocab is None:
                # Infer from embedding weight shape
                ckpt_vocab = checkpoint['model_state_dict'].get(
                    'encoder.embedding.weight', torch.empty(0)
                ).shape[0]
            
            if ckpt_vocab > 0 and ckpt_vocab < VOCAB_SIZE:
                print(f"   -> Auto-resizing embeddings: {ckpt_vocab} → {VOCAB_SIZE}")
                for key in ['encoder.embedding.weight', 'decoder.embedding.weight', 'decoder.fc_out.weight']:
                    if key in model.state_dict() and model.state_dict()[key].shape[0] == VOCAB_SIZE:
                        old_w = checkpoint['model_state_dict'].get(key)
                        if old_w is not None and old_w.shape[0] == ckpt_vocab:
                            mean_emb = old_w.mean(dim=0, keepdim=True)
                            num_new = VOCAB_SIZE - ckpt_vocab
                            new_rows = mean_emb.expand(num_new, old_w.shape[1]).clone()
                            checkpoint['model_state_dict'][key] = torch.cat([old_w, new_rows], dim=0)
                # Reload with resized weights
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"   -> Embedding resize complete.")

            start_global_epoch = checkpoint.get('global_epoch', 0)
            start_chunk_idx = checkpoint.get('chunk_idx', 0)
            global_step = checkpoint.get('global_step', 0)
            
            # Detect intermediate checkpoints (either every 1000 batches or the old mid_chunk style)
            is_intermediate = "batch" in os.path.basename(latest_checkpoint) or "mid_chunk" in os.path.basename(latest_checkpoint)
            
            if is_intermediate:
                start_batch_idx = checkpoint.get('batch_idx', -1)
                if start_batch_idx != -1:
                    print(f"   -> Intermediate Checkpoint Detected! Dataloader will fast-forward to Batch {start_batch_idx + 1}")
                else:
                    print(f"   -> Old Intermediate Save Detected (No Batch Info). Will re-run this chunk from Batch 0.")
            else:
                start_chunk_idx += 1
                global_step += 1
                if start_chunk_idx >= TOTAL_CHUNKS:
                    start_chunk_idx = 0
                    start_global_epoch += 1

            if start_global_epoch >= GLOBAL_EPOCHS:
                print("Training is already complete for all epochs!")
                return

            print(f"Resume Training -> Global Epoch: {start_global_epoch+1}, Chunk Index: {start_chunk_idx+1}")
        else:
            print("No Checkpoint found. Starting New Training.")

# ==========================================
# 4. Multi-Epoch Chunk Training Loop
# ==========================================
    print("\n" + "="*50)
    print(f"START TRAINING ({GLOBAL_EPOCHS} TOTAL GLOBAL EPOCHS)")
    print("="*50)

    for epoch in range(start_global_epoch, GLOBAL_EPOCHS):
        local_epoch_display = epoch + 1
               
        for chunk_idx in range(start_chunk_idx, TOTAL_CHUNKS):
            current_file = chunk_files[chunk_idx]
            
            total_lines = count_lines(current_file)
            # Each line produces 2 training pairs (EN→JA + JA→EN)
            total_batches_in_file = math.ceil((total_lines * 2) / BATCH_SIZE)
            
            print(f" [Processing File: {os.path.basename(current_file)}]")
            print(f" [Local Epoch: {local_epoch_display}/{GLOBAL_EPOCHS}]")
            print(f" [Total Batches: {total_batches_in_file:,} | Saving every 1000 batches]")
            
            train_loader, _ = get_dataloader(
                filepath=current_file, src_tok_path=SRC_TOKENIZER,
                tgt_tok_path=TGT_TOKENIZER, batch_size=BATCH_SIZE, min_score=0.70
            )

            model.train()
            total_loss = 0
            batch_count = 0
            optimizer.zero_grad() 

            for batch_idx, (src, tgt) in enumerate(train_loader):
                
                # Fast-forward Dataloader if resuming from an intermediate checkpoint
                if start_batch_idx != -1:
                    if batch_idx <= start_batch_idx:
                        if batch_idx % 1000 == 0:
                            print(f"Fast-forwarding Dataloader... skipping batch {batch_idx}/{start_batch_idx}")
                        continue
                    else:
                        print(f"\n Resume training successfully at Batch {batch_idx}!\n")
                        start_batch_idx = -1 

                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                tgt_input    = tgt[:, :-1]
                tgt_expected = tgt[:, 1:]

                with autocast(device_type='cuda'):
                    output = model(src, tgt_input)
                    output_reshaped = output.reshape(-1, output.shape[-1])
                    tgt_expected_reshaped = tgt_expected.reshape(-1)
                    
                    loss = criterion(output_reshaped, tgt_expected_reshaped)
                    loss = loss / ACCUMULATION_STEPS

                scaler.scale(loss).backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += (loss.item() * ACCUMULATION_STEPS)
                batch_count += 1

                # ==========================================
                # Print Progress
                # ==========================================
                # Print immediately after a 1000-batch save (e.g., at batch 1001, 2001)
                just_saved = (batch_idx > 1 and (batch_idx - 1) % 1000 == 0)
                
                # Print immediately upon resuming from a checkpoint
                just_resumed = False
                if RESUME_TRAINING and 'checkpoint' in locals() and 'is_intermediate' in locals():
                    if is_intermediate:
                        resume_batch = checkpoint.get('batch_idx', -1)
                        if resume_batch != -1 and batch_idx == resume_batch + 1:
                            just_resumed = True

                is_first_batch = (batch_idx == 1)
                
                # Condition: Print every 200 batches (excluding batch 0), OR right after resume/save
                if batch_idx > 0 and (batch_idx % 200 == 0 or just_resumed or just_saved or is_first_batch):
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  [Epoch {local_epoch_display:02d} | Chunk {chunk_idx+1:02d}] Batch {batch_idx:04d}/~{total_batches_in_file} | Loss: {(loss.item()*ACCUMULATION_STEPS):.4f} | LR: {current_lr:.7f}")

                # ==========================================
                # Save Intermediate Checkpoint
                # ==========================================
                # Save exactly at every 1000 batches
                if batch_idx % 1000 == 0 and batch_idx > 0:
                    inter_loss = total_loss / batch_count
                    checkpoint_name = f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_batch_{batch_idx}.pt"
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
                    
                    torch.save({
                        'global_epoch': epoch,
                        'chunk_idx': chunk_idx,
                        'batch_idx': batch_idx,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': inter_loss,
                    }, checkpoint_path)
                    print(f"\nSaved Checkpoint: {checkpoint_name}\n")

                    # Clean up the previous intermediate checkpoint (1000 batches ago) to save space
                    prev_batch_idx = batch_idx - 1000
                    if prev_batch_idx > 0:
                        old_inter = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_batch_{prev_batch_idx}.pt")
                        if os.path.exists(old_inter):
                            os.remove(old_inter)

            # ==========================================
            # End of Chunk Operations
            # ==========================================
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            print("-" * 50)
            print(f"SUMMARY: Finished Local_Epoch [{local_epoch_display}/{GLOBAL_EPOCHS}] for {os.path.basename(current_file)}")
            print(f"   Average Loss: {avg_loss:.4f}")
            print("-" * 50)

            scheduler.step()
            global_step += 1

            del train_loader
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

            # Save the final checkpoint for the completed chunk
            checkpoint_name = f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}.pt"
            checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
            torch.save({
                'global_epoch': epoch,
                'chunk_idx': chunk_idx,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved Checkpoint: {checkpoint_name}\n")

            # Clean up ALL intermediate batch checkpoints for this chunk once the full chunk is saved
            old_inters = glob.glob(os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_batch_*.pt"))
            for old_file in old_inters:
                try:
                    os.remove(old_file)
                except OSError:
                    pass
            
            # Clean up old mid_chunk files (backward compatibility with previous version)
            old_mid = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_mid_chunk.pt")
            if os.path.exists(old_mid):
                os.remove(old_mid)

        # Reset chunk index for the next global epoch
        start_chunk_idx = 0

    print("\nTraining completed for all epochs!")

if __name__ == "__main__":
    train()