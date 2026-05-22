import torch
import torch.nn as nn
import math
import glob
from torch.optim import Adam
from torch.amp import GradScaler, autocast
import os
# Change StepLR to LambdaLR for Warmup + Cosine Decay
from torch.optim.lr_scheduler import LambdaLR

from model.dataloader import get_dataloader
from model.transformer import Transformer

# ==========================================
# Hyperparameters Configuration
# ==========================================
BATCH_SIZE = 32
TOTAL_EPOCHS = 20          
LEARNING_RATE = 1e-3
MAX_LENGTH = 128


# Model Configuration
D_MODEL = 256
NUM_LAYERS = 6
NUM_HEADS = 8
D_FF = 1024

# Warmup + Cosine scheduler
WARMUP_EPOCHS = 2          # gradually increase LR in two epochs, Loss up fix

# dataset path
TRAIN_FILE = "data/dataset/300Ktrain2_clean.txt"
SRC_TOKENIZER = "data/shared_tokenizer.json"
TGT_TOKENIZER = "data/shared_tokenizer.json"

# checkpoints setting
RESUME_TRAINING = True
CHECKPOINT_DIR = "checkpoints"

# check gpu
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    print(f"Preparing for Training on device: {DEVICE}")

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

# ==========================================
# 1. Data Preparation
# ==========================================
    print("Downloading Dataset...")
    train_loader, pad_idx = get_dataloader(
        filepath=TRAIN_FILE,
        src_tok_path=SRC_TOKENIZER,
        tgt_tok_path=TGT_TOKENIZER,
        batch_size=BATCH_SIZE,
        min_score=0.70
    )

    VOCAB_SIZE = 32000

# ==========================================
# 2. Model, Loss, Optimizer
# ==========================================
    print("Creating Transformer Model...")
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        src_pad_idx=pad_idx,
        tgt_pad_idx=pad_idx,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        dropout=0.1
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # --- Warmup + Cosine Decay ---
    # Use Lambda (fix loss)
    # Warmup: LR increase from 0 → LEARNING_RATE in WARMUP_EPOCHS 
    # Cosine: LR smooth decreaase until last epoch
    def lr_lambda(current_epoch):
        if current_epoch < WARMUP_EPOCHS:
            return (current_epoch + 1) / WARMUP_EPOCHS
        progress = (current_epoch - WARMUP_EPOCHS) / max(1, TOTAL_EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = GradScaler(device='cuda')

# ==========================================
# 3. Checkpoint Load
# ==========================================
    start_epoch = 0
    latest_checkpoint = None

    if RESUME_TRAINING:
        # search transformer_epoch_*.pt file
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, "transformer_epoch_*.pt"))
        
        if checkpoint_files:
            # find latest epoch
            latest_checkpoint = max(checkpoint_files, key=lambda f: int(f.split('_epoch_')[-1].split('.pt')[0]))

    if latest_checkpoint and os.path.exists(latest_checkpoint):
        print(f"Loading Checkpoint From: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, weights_only=True)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch']
        print(f"Loading Completed! Continue Training at Epoch no.{start_epoch + 1}")
    else:
        print("No Checkpoint file found, Start New Training")

# ==========================================
# 4. Training Loop
# ==========================================
    print("Start Training")
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        loss = None  # fix loss unbound if dataloader is none

        for batch_idx, (src, tgt) in enumerate(train_loader):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            # --- Teacher Forcing ---
            tgt_input    = tgt[:, :-1]   # remove </s> 
            tgt_expected = tgt[:, 1:]    # remove <s> 

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                output = model(src, tgt_input)
                output_reshaped       = output.reshape(-1, output.shape[-1])
                tgt_expected_reshaped = tgt_expected.reshape(-1)
                loss = criterion(output_reshaped, tgt_expected_reshaped)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            batch_count += 1

            if batch_idx % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch [{epoch+1}/{TOTAL_EPOCHS}] | Batch {batch_idx:04d} | Loss: {loss.item():.4f} | LR: {current_lr:.7f}")

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f"=== Finish Epoch {epoch+1} | Avg Loss: {avg_loss:.4f} ===")

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate For The Next Epoch: {current_lr:.7f}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"transformer_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': loss.item() if loss is not None else None,
        }, checkpoint_path)
        print(f"Saved Model at: {checkpoint_path}\n")

if __name__ == "__main__":
    train()