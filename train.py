import torch
import torch.nn as nn
import math
import glob
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
import os
import gc
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
VOCAB_SIZE = 50000

CHUNK_DIR = "data/dataset/chunks"
SRC_TOKENIZER = "data/tokenizer/shared_tokenizer.json"
TGT_TOKENIZER = "data/tokenizer/shared_tokenizer.json"

RESUME_TRAINING = True
CHECKPOINT_DIR = "checkpoints"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def count_lines(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def train():
    print(f"Prepare for Training on device: {DEVICE}")

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

    print("Create Transformer Model...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE, tgt_vocab_size=VOCAB_SIZE,
        src_pad_idx=pad_idx, tgt_pad_idx=pad_idx,
        d_model=D_MODEL, num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS, d_ff=D_FF, dropout=0.1
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
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
    
    # [NEW] Variable to track exactly which batch we stopped at
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
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            start_global_epoch = checkpoint.get('global_epoch', 0)
            start_chunk_idx = checkpoint.get('chunk_idx', 0)
            global_step = checkpoint.get('global_step', 0)
            
            # [FIXED] Detect if it's a Mid-Chunk save and prepare to fast-forward
            if "mid_chunk" in os.path.basename(latest_checkpoint):
                start_batch_idx = checkpoint.get('batch_idx', -1)
                if start_batch_idx != -1:
                    print(f"   -> Mid-Chunk Detected Dataloader will fast-forward to Batch {start_batch_idx + 1}")
                else:
                    print(f"   -> Old Mid-Chunk Save Detected (No Batch Info). Will re-run this chunk from Batch 0.")
            else:
                # Normal End-of-Chunk resume logic
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
        
        print(f"GLOBAL_EPOCH [{epoch+1}/{GLOBAL_EPOCHS}]")
        
        for chunk_idx in range(start_chunk_idx, TOTAL_CHUNKS):
            current_file = chunk_files[chunk_idx]
            
            total_lines = count_lines(current_file)
            total_batches_in_file = math.ceil(total_lines / BATCH_SIZE)
            midpoint_batch = total_batches_in_file // 2
            
            print(f"\n   Processing File: {os.path.basename(current_file)}")
            print(f"     [Trainset info: {total_lines:,} lines | Total batch: {total_batches_in_file:,} batches | Midpoint saving at batch: {midpoint_batch:,}]")
            
            train_loader, _ = get_dataloader(
                filepath=current_file, src_tok_path=SRC_TOKENIZER,
                tgt_tok_path=TGT_TOKENIZER, batch_size=BATCH_SIZE, min_score=0.70
            )

            model.train()
            total_loss = 0
            batch_count = 0
            optimizer.zero_grad() 

            for batch_idx, (src, tgt) in enumerate(train_loader):
                
                # [NEW] Fast-forward mechanism: Skip batches if we resumed from a mid-chunk
                if start_batch_idx != -1:
                    if batch_idx <= start_batch_idx:
                        if batch_idx % 1000 == 0:
                            print(f"Fast-forwarding Dataloader... skipping batch {batch_idx}/{start_batch_idx}")
                        continue  # Skip training for this batch completely
                    else:
                        print(f"\n Resume training successful at Batch {batch_idx}!\n")
                        start_batch_idx = -1  # Reset so it doesn't affect future chunks

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

                # [UPDATED] Print progress every 200 batches OR immediately after resuming
                if batch_idx % 200 == 0 or batch_idx == (checkpoint.get('batch_idx', -1) if RESUME_TRAINING and "mid_chunk" in os.path.basename(latest_checkpoint) else -1) + 1:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"  [Epoch {local_epoch_display:02d} | Chunk {chunk_idx+1:02d}] Batch {batch_idx:04d} | Loss: {(loss.item()*ACCUMULATION_STEPS):.4f} | LR: {current_lr:.7f}")


                # [UPDATED] Save the exact batch index so the fast-forward knows where to stop next time
                if batch_idx == midpoint_batch and batch_idx > 0:
                    mid_loss = total_loss / batch_count
                    checkpoint_name = f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_mid_chunk.pt"
                    checkpoint_path = os.path.join(CHECKPOINT_DIR, checkpoint_name)
                    
                    torch.save({
                        'global_epoch': epoch,
                        'chunk_idx': chunk_idx,
                        'batch_idx': batch_idx,  # <--- FIXED: Now saves the exact step!
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'loss': mid_loss,
                    }, checkpoint_path)
                    print(f"\n [MIDPOINT REACHED] Saved Mid-Chunk Checkpoint: {checkpoint_name}\n")

            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            
            print("-" * 50)
            print(f"SUMMARY: Finished Epoch [{local_epoch_display}/{GLOBAL_EPOCHS}] for {os.path.basename(current_file)}")
            print(f"   Average Loss: {avg_loss:.4f}")
            print("-" * 50)

            scheduler.step()
            global_step += 1

            del train_loader
            gc.collect()

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

            old_mid = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}_chunk_{chunk_idx+1}_mid_chunk.pt")
            if os.path.exists(old_mid):
                os.remove(old_mid)

        start_chunk_idx = 0

    print("\nTraining completed for all epochs!")

if __name__ == "__main__":
    train()