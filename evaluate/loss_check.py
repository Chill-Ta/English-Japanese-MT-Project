import torch
import os
import glob
import matplotlib.pyplot as plt

# Directory where our checkpoint files are stored
CHECKPOINT_DIR = "checkpoints"

def plot_training_loss():
    print(f"Searching for checkpoint files in directory '{CHECKPOINT_DIR}'...")
    
    # Find all .pt files in the directory
    pt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "transformer_epoch_*.pt"))
    
    if not pt_files:
        print("No checkpoint files found.")
        return

    epoch_loss_data = []

    # Loop through and read data from each file
    for file in pt_files:
        try:
            # Use map_location='cpu' to avoid consuming GPU VRAM just for viewing the data
            checkpoint = torch.load(file, map_location='cpu')
            
            # Extract Epoch and Loss values
            if 'epoch' in checkpoint and 'loss' in checkpoint:
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']
                epoch_loss_data.append((epoch, loss))
            
        except Exception as e:
            print(f"⚠️ Unable to read file {file}: {e}")

    # Sort data by Epoch in ascending order
    epoch_loss_data.sort(key=lambda x: x[0])

    # Unpack data to prepare for plotting
    epochs = [data[0] for data in epoch_loss_data]
    losses = [data[1] for data in epoch_loss_data]

    # --- 1. Display summary in the terminal ---
    print("\n" + "="*30)
    print("Average Loss Summary")
    print("="*30)
    for ep, ls in epoch_loss_data:
        print(f"Epoch {ep:02d} | Loss: {ls:.4f}")
    print("="*30)

    # --- 2. Generate Plot (Graph) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    plt.title('Training Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Average Loss', fontsize=14)
    plt.xticks(epochs) # Force X-axis to show every Epoch
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot as an image
    #plt.savefig('loss_curve.png', bbox_inches='tight')
    #print("\nPlot generated successfully! Saved as 'loss_curve.png'.")
    
    # Display the plot on screen
    plt.show()

if __name__ == "__main__":
    plot_training_loss()