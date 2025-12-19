import os
import argparse
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MarmosetDataset
from model import WaveNetSourceSeparator

def save_sample_spectrograms(dataset, save_dir, num_samples=5):
    """Save spectrograms and waveforms of random samples from the dataset."""
    os.makedirs(save_dir, exist_ok=True)
    indices = torch.randperm(len(dataset))[:num_samples]
    
    print(f"Saving {num_samples} sample spectrograms/waveforms to {save_dir}...")
    
    for i, idx in enumerate(indices):
        mixture, target = dataset[idx]
        # mixture and target are tensors, likely (1, T)
        
        # Calculate residual (the other call)
        residual = mixture - target
        
        # Plot: 6 rows (Spectrogram, Waveform for each of Mixture, Target, Residual)
        fig, axes = plt.subplots(6, 1, figsize=(10, 18))
        
        def plot_group(spec_ax, wave_ax, data, title):
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            data = data.flatten()
            
            # Create time axis
            times = np.arange(len(data)) / dataset.sample_rate
            
            # Spectrogram
            spec_ax.specgram(data, Fs=dataset.sample_rate, NFFT=512, noverlap=256, cmap='viridis')
            spec_ax.set_title(f"{title} (Spectrogram)")
            spec_ax.set_ylabel("Frequency (Hz)")
            spec_ax.set_xticks([]) # Remove x-ticks for spectrogram to reduce clutter
            
            # Waveform
            wave_ax.plot(times, data, linewidth=0.5, alpha=0.8)
            wave_ax.set_title(f"{title} (Waveform)")
            wave_ax.set_ylabel("Amplitude")
            wave_ax.set_xlim(0, times[-1])
            wave_ax.grid(True, alpha=0.3)
        
        plot_group(axes[0], axes[1], mixture, "Mixture")
        plot_group(axes[2], axes[3], target, "Target (Call 1)")
        plot_group(axes[4], axes[5], residual, "Residual (Call 2)")
        
        axes[5].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{i+1}.png"))
        plt.close(fig)

def train(data_dir):
    # Hyperparameters
    # data_dir argument passed from command line
    batch_size = 8
    num_epochs = 100
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Datasets
    train_dataset = MarmosetDataset(data_dir, segment_length=16384, split='train')
    val_dataset = MarmosetDataset(data_dir, segment_length=16384, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Visualize training samples
    save_sample_spectrograms(train_dataset, "images", num_samples=5)

    # Model
    model = WaveNetSourceSeparator(in_channels=1, out_channels=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training on {device}...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Track history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for i, (mixture, target) in enumerate(train_loader):
            mixture = mixture.to(device)
            target = target.to(device)
            
            # Forward pass
            output = model(mixture)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for mixture, target in val_loader:
                mixture = mixture.to(device)
                target = target.to(device)
                
                output = model(mixture)
                loss = criterion(output, target)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth"))

    print("Training complete.")
    torch.save(model.state_dict(), "model_final.pth")
    
    # Save training history
    import pickle
    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("Training history saved to training_history.pkl")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the WaveNet source separator.')
    parser.add_argument('data_dir', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('history_fname', type=str, help='File name for the training history file.')
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
        
    history = train(args.data_dir)
    with open(args.history_fname, 'wb') as f:
        pickle.dump(history, f)
