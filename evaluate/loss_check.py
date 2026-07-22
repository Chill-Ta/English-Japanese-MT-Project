import torch
import os
import glob
import re
import matplotlib.pyplot as plt

# Directory where our checkpoint files are stored
CHECKPOINT_DIR = "checkpoints"

def plot_training_loss():
    print(f"Searching for checkpoint files in directory '{CHECKPOINT_DIR}'...")
    
    # [FIX 1] Updated glob pattern to match our new checkpoint naming standard
    pt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "transformer_epoch_*_chunk_*.pt"))
    
    if not pt_files:
        print("No checkpoint files found.")
        return

    epoch_to_max_chunk = {}

    # First, scan all checkpoint files and identify the last chunk index for each epoch
    for file in pt_files:
        filename = os.path.basename(file)
        match = re.search(r"transformer_epoch_(\d+)_chunk_(\d+)\.pt", filename)
        if match:
            file_epoch = int(match.group(1))
            file_chunk = int(match.group(2))
            
            # Keep track of the file with the largest chunk index for each epoch
            if file_epoch not in epoch_to_max_chunk or file_chunk > epoch_to_max_chunk[file_epoch][0]:
                epoch_to_max_chunk[file_epoch] = (file_chunk, file)

    epoch_loss_data = []

    # Loop through only the selected last chunk of each epoch
    for file_epoch in sorted(epoch_to_max_chunk.keys()):
        file_chunk, file = epoch_to_max_chunk[file_epoch]
        try:
            # Use map_location='cpu' to avoid consuming GPU VRAM just for viewing the data
            checkpoint = torch.load(file, map_location='cpu', weights_only=True)
            
            # [FIX 2] Updated key name from 'epoch' to 'global_epoch' based on train.py
            if 'global_epoch' in checkpoint and 'loss' in checkpoint:
                loss = checkpoint['loss']
                
                # Store data along with filename tokens for accurate time-series plotting
                epoch_loss_data.append((file_epoch, file_chunk, loss))
            
        except Exception as e:
            print(f"Unable to read file {file}: {e}")

    # Sort data chronologically by Epoch first, then by Chunk index
    epoch_loss_data.sort(key=lambda x: (x[0], x[1]))

    # Unpack data to prepare for plotting
    # We create a timeline string like "E1-C1", "E1-C2" for the X-axis labels
    timeline_labels = [f"E{data[0]}-C{data[1]}" for data in epoch_loss_data]
    losses = [data[2] for data in epoch_loss_data]

    # --- 1. Display summary in the terminal ---
    print("\n" + "="*45)
    print("      Average Loss Progress Summary")
    print("="*45)
    for ep, ch, ls in epoch_loss_data:
        print(f"Epoch {ep:02d} | Chunk {ch:02d} | Avg Loss: {ls:.4f}")
    print("="*45)

    # --- 2. Generate Plot (Graph) ---
    plt.figure(figsize=(12, 6))
    
    # Plotting the timeline trend
    plt.plot(timeline_labels, losses, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    plt.title('Training Loss Curve (Chronological Progress)', fontsize=16)
    plt.xlabel('Timeline (E = Global Epoch, C = Chunk File)', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    
    # Rotate labels to prevent them from overlapping if you have many chunks
    plt.xticks(rotation=45, ha='right', fontsize=9) 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the plot as an image automatically so you can view it outside terminal
    # plt.savefig('loss_curve.png', bbox_inches='tight')
    # print("\nPlot generated successfully! Saved as 'loss_curve.png'.")
    
    # Display the plot on screen
    plt.show()

if __name__ == "__main__":
    plot_training_loss()