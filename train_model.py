import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import json


# ============================================================================
# CONFIGURATION
# ============================================================================

class TrainConfig:
    
    # Model architecture (from paper)
    # Input: (batch, 6, 360, T) - 6 harmonic channels
    # Output: (batch, 1, 360, T) - salience representation
    
    # Training parameters
    LEARNING_RATE = 0.001
    BATCH_SIZE = 1  # Variable length tracks
    NUM_EPOCHS = 50
    
    # Optimizer
    OPTIMIZER = 'adam'  # Paper uses Adam
    
    # Loss function
    LOSS_FUNCTION = 'binary_cross_entropy'  # Paper uses cross-entropy
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    MODEL_SAVE_DIR = Path("Data/models")
    CHECKPOINT_DIR = Path("Data/checkpoints")
    RESULTS_DIR = Path("Data/results")
    
    # Early stopping
    PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs
    
    # Save frequency
    SAVE_EVERY = 5  # Save checkpoint every 5 epochs
    RESUME_CHECKPOINT = "Data/checkpoints/checkpoint_epoch_17.pt"  # Change to your checkpoint

# ============================================================================
# CNN ARCHITECTURE (from Paper Figure 2)
# ============================================================================

class DeepSalienceCNN(nn.Module):
    """
    Fully Convolutional Neural Network for Pitch Salience Estimation.
    
    Architecture from paper:
    - Layer 1: 128 filters (5x5)
    - Layer 2: 64 filters (5x5)
    - Layer 3: 64 filters (3x3)
    - Layer 4: 64 filters (3x3)
    - Layer 5: 8 filters (70x3)
    - Output: 1 filter (1x1) with sigmoid
    
    All layers use:
    - Batch normalization
    - ReLU activation (except final layer uses sigmoid)
    - Zero padding to maintain spatial dimensions
    """
    
    def __init__(self):
        super(DeepSalienceCNN, self).__init__()
        
        # Layer 1: 6 -> 128 channels, 5x5 kernel
        # Covers ~1 semitone in frequency, ~50ms in time
        self.conv1 = nn.Conv2d(
            in_channels=6,      # 6 harmonic channels
            out_channels=128,
            kernel_size=(5, 5),
            padding=(2, 2)      # Zero padding to maintain shape
        )
        self.bn1 = nn.BatchNorm2d(128)
        
        # Layer 2: 128 -> 64 channels, 5x5 kernel
        self.conv2 = nn.Conv2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(5, 5),
            padding=(2, 2)
        )
        self.bn2 = nn.BatchNorm2d(64)
        
        # Layer 3: 64 -> 64 channels, 3x3 kernel
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn3 = nn.BatchNorm2d(64)
        
        # Layer 4: 64 -> 64 channels, 3x3 kernel
        self.conv4 = nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            padding=(1, 1)
        )
        self.bn4 = nn.BatchNorm2d(64)
        
        # Layer 5: 64 -> 8 channels, 70x3 kernel
        # 70 bins in frequency = ~14 semitones (captures octave relationships)
        # For kernel_size=70 (even), we need padding=35 on each side to maintain size
        # But PyTorch's padding is symmetric, so we get output_size = input + 1
        # Solution: Use padding=34 which gives 359, then we'll crop in forward()
        self.conv5 = nn.Conv2d(
            in_channels=64,
            out_channels=8,
            kernel_size=(70, 3),
            padding=(35, 1)     # This gives 361, we'll crop to 360 in forward
        )
        self.bn5 = nn.BatchNorm2d(8)
        
        # Output layer: 8 -> 1 channel, 1x1 kernel
        self.conv_out = nn.Conv2d(
            in_channels=8,
            out_channels=1,
            kernel_size=(1, 1),
            padding=(0, 0)
        )
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass.
        
        Input shape: (batch, 6, 360, T)
        Output shape: (batch, 1, 360, T)
        """
        # Layer 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        # Layer 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Layer 5
        x = self.conv5(x)
        print(f"DEBUG: After conv5, shape = {x.shape}")  # Debug
        # Crop to 360 bins BEFORE batch norm (conv5 with kernel=70 gives 361)
        if x.shape[2] == 361:
            x = x[:, :, :-1, :]  # Remove last frequency bin to get 360
            print(f"DEBUG: After crop, shape = {x.shape}")  # Debug
        x = self.bn5(x)
        x = self.relu(x)
        
        # Output layer with sigmoid
        x = self.conv_out(x)
        x = self.sigmoid(x)
        print(f"DEBUG: Final output shape = {x.shape}")  # Debug
        
        return x


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, criterion, optimizer, device, config):
    """
    Train model for one epoch.
    
    Training uses random segments (5 seconds) so they always fit in memory.
    
    Returns:
        avg_loss: Average training loss for this epoch
    """
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training")
    
    for batch in pbar:
        # Get data - segments are small, always fit in memory
        hcqt = batch['hcqt'].to(device)      # (batch, 6, 360, ~430)
        target = batch['target'].to(device)  # (batch, 360, ~430)
        
        # Add channel dimension to target
        target = target.unsqueeze(1)  # (batch, 1, 360, ~430)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(hcqt)  # (batch, 1, 360, ~430)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track loss
        running_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = running_loss / num_batches
    return avg_loss


def validate(model, dataloader, criterion, device, config):
    """
    Validate model on validation set.
    
    Validation uses full tracks, processed in sliding windows if needed.
    
    Returns:
        avg_loss: Average validation loss
    """
    from data_set_prep import Config as DataConfig
    
    model.eval()
    running_loss = 0.0
    num_batches = 0
    
    # Segment size for sliding window (use data config)
    data_config = DataConfig()
    segment_frames = int(
        data_config.SEGMENT_LENGTH * data_config.SAMPLE_RATE / data_config.HOP_LENGTH
    )
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # Get data
            hcqt = batch['hcqt']      # (1, 6, 360, T) - full track
            target = batch['target']  # (1, 360, T)
            target = target.unsqueeze(1)  # (1, 1, 360, T)
            
            batch_size, channels, freq_bins, time_frames = hcqt.shape
            
            # If track is longer than segment, process in sliding windows
            if time_frames > segment_frames:
                step_size = segment_frames // 2  # 50% overlap
                outputs = []
                targets_list = []
                
                for start in range(0, time_frames, step_size):
                    end = min(start + segment_frames, time_frames)
                    
                    # If last segment is too short, adjust start
                    if end - start < segment_frames // 2:
                        break
                    
                    # Extract window
                    hcqt_window = hcqt[:, :, :, start:end].to(device)
                    target_window = target[:, :, :, start:end].to(device)
                    
                    # Forward pass
                    output_window = model(hcqt_window)
                    
                    outputs.append(output_window.cpu())
                    targets_list.append(target_window.cpu())
                
                # Compute loss on all windows
                window_losses = []
                for out, tgt in zip(outputs, targets_list):
                    out = out.to(device)
                    tgt = tgt.to(device)
                    loss = criterion(out, tgt)
                    window_losses.append(loss.item())
                
                batch_loss = np.mean(window_losses)
            else:
                # Short track, process directly
                hcqt = hcqt.to(device)
                target = target.to(device)
                
                # Forward pass
                output = model(hcqt)
                
                # Compute loss
                loss = criterion(output, target)
                batch_loss = loss.item()
            
            # Track loss
            running_loss += batch_loss
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': batch_loss})
    
    avg_loss = running_loss / num_batches
    return avg_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, config, is_best=False):
    """Save model checkpoint"""
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    # Save regular checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model
    if is_best:
        best_path = config.CHECKPOINT_DIR / "best_model.pt"
        torch.save(checkpoint, best_path)
        print(f"Saved best model (val_loss: {val_loss:.4f})")


def plot_losses(train_losses, val_losses, config):
    """Plot and save training curves"""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = config.RESULTS_DIR / "training_curves.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved training curves to {save_path}")


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train_model(dataloaders, config=TrainConfig()):
    """
    Main training function.
    
    Parameters:
        dataloaders: Dict with 'train' and 'val' DataLoaders
        config: Training configuration
    """
    print("\n" + "="*60)
    print("TRAINING DEEP SALIENCE CNN")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print("="*60 + "\n")
    
    # Create directories
    config.MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = DeepSalienceCNN().to(config.DEVICE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss function (Binary Cross-Entropy as in paper)
    criterion = nn.BCELoss()
    
    # Optimizer (Adam as in paper)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, config.DEVICE, config
        )
        
        # Validate
        val_loss = validate(
            model, dataloaders['val'], criterion, config.DEVICE, config
        )
        
        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        
        # Check if best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint
        if epoch % config.SAVE_EVERY == 0 or is_best:
            save_checkpoint(
                model, optimizer, epoch, train_loss, val_loss, config, is_best
            )
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Save final model
    final_model_path = config.MODEL_SAVE_DIR / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\nSaved final model to {final_model_path}")
    
    # Plot training curves
    plot_losses(train_losses, val_losses, config)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'config': {
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'num_epochs': config.NUM_EPOCHS
        }
    }
    
    history_path = config.RESULTS_DIR / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {final_model_path}")
    print(f"Training curves: {config.RESULTS_DIR / 'training_curves.png'}")
    
    return model, history


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Train the model using prepared data.
    
    Usage:
        python train_model.py
    
    Make sure to run data_set_prep.py first to prepare the data.
    """
    
    # Import data preparation
    from data_set_prep import prepare_data, Config as DataConfig
    
    # Prepare data
    print("Loading data...")
    dataloaders, splits = prepare_data(DataConfig())
    
    # Train model
    config = TrainConfig()
    model, history = train_model(dataloaders, config)
    
    print("\nTraining complete! Next steps:")
    print("1. Check training curves in Data/results/")
    print("2. Evaluate on validation set")
    print("3. Run final evaluation on Evaluation/vocadito")
