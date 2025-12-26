import os
import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np
import torch
import soundfile as sf
from dataset import MarmosetDataset

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
        

def plot_spectrogram(waveform, sample_rate, title="Spectrogram"):
    """Plot spectrogram of a waveform."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.flatten()
    
    # Compute spectrogram
    plt.figure(figsize=(10, 4))
    plt.specgram(waveform, Fs=sample_rate, NFFT=512, noverlap=256, cmap='viridis')
    plt.colorbar(label='Intensity (dB)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def visualize_clean_vocalization(data_dir, file_idx=0, save_path=None):
    """Visualize a single clean vocalization with spectrogram and waveform."""
    dataset = MarmosetDataset(data_dir, split='train')
    
    # Load a single file directly
    filepath = dataset.file_list[file_idx]
    waveform = dataset.load_and_crop(filepath)
    
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.numpy()
    
    if waveform.ndim > 1:
        waveform = waveform.flatten()
        
    times = np.arange(len(waveform)) / dataset.sample_rate
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Spectrogram
    axes[0].specgram(waveform, Fs=dataset.sample_rate, NFFT=512, noverlap=256, cmap='viridis')
    axes[0].set_title(f"Clean Vocalization: {filepath.split('/')[-1]} (Spectrogram)")
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xticks([]) # Remove x-ticks
    
    # Waveform
    axes[1].plot(times, waveform, linewidth=0.5, alpha=0.8)
    axes[1].set_title(f"Clean Vocalization: {filepath.split('/')[-1]} (Waveform)")
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_xlim(0, times[-1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_mixture(data_dir, sample_idx=0, save_path=None):
    """Visualize a mixture and its components."""
    dataset = MarmosetDataset(data_dir, split='train')
    
    # Get a mixture sample
    mixture, target = dataset[sample_idx]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Mixture spectrogram
    spec, freqs, t, im = axes[0].specgram(mixture.numpy().flatten(), Fs=dataset.sample_rate, 
                     NFFT=512, noverlap=256, cmap='viridis')
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_title('Mixture (voc1 + voc2)')
    
    # Target (voc1) spectrogram
    axes[1].specgram(target.numpy().flatten(), Fs=dataset.sample_rate, 
                     NFFT=512, noverlap=256, cmap='viridis')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_title('Target (voc1)')
    
    # Residual (approximate voc2) spectrogram
    residual = mixture - target
    axes[2].specgram(residual.numpy().flatten(), Fs=dataset.sample_rate, 
                          NFFT=512, noverlap=256, cmap='viridis')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Residual (voc2)')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, label='Intensity (dB)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_multiple_mixtures(data_dir, num_samples=4, save_path=None):
    """Visualize multiple mixture examples."""
    dataset = MarmosetDataset(data_dir, split='train')
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3*num_samples))
    
    for i in range(num_samples):
        mixture, target = dataset[i]
        
        # Mixture
        axes[i, 0].specgram(mixture.numpy().flatten(), Fs=dataset.sample_rate, 
                           NFFT=512, noverlap=256, cmap='viridis')
        axes[i, 0].set_ylabel('Frequency (Hz)')
        axes[i, 0].set_title(f'Mixture {i+1}')
        
        # Target
        axes[i, 1].specgram(target.numpy().flatten(), Fs=dataset.sample_rate, 
                           NFFT=512, noverlap=256, cmap='viridis')
        axes[i, 1].set_ylabel('Frequency (Hz)')
        axes[i, 1].set_title(f'Target {i+1}')
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_file(filepath, save_path=None):
    """Visualize a specific audio file with spectrogram and waveform."""
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found.")
        return

    # Load audio
    audio, sample_rate = sf.read(filepath)
    
    # Handle stereo/mono and shape
    if audio.ndim > 1:
        # Convert to mono for visualization if stereo
        audio = np.mean(audio, axis=1)
    
    waveform = audio
    times = np.arange(len(waveform)) / sample_rate
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Spectrogram
    axes[0].specgram(waveform, Fs=sample_rate, NFFT=512, noverlap=256, cmap='viridis')
    axes[0].set_title(f"File: {os.path.basename(filepath)} (Spectrogram)")
    axes[0].set_ylabel('Frequency (Hz)')
    axes[0].set_xticks([]) # Remove x-ticks
    
    # Waveform
    axes[1].plot(times, waveform, linewidth=0.5, alpha=0.8)
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_xlim(0, times[-1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_directory(input_dir, output_dir=None):
    """Visualize all audio files in a directory."""
    if not os.path.exists(input_dir):
        print(f"Error: Directory {input_dir} not found.")
        return

    # Find all audio files
    extensions = ['*.wav', '*.flac', '*.mp3']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not files:
        print(f"No audio files found in {input_dir}")
        return
        
    print(f"Found {len(files)} audio files in {input_dir}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = "images"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to {output_dir}...")
    
    for i, filepath in enumerate(files):
        filename = os.path.basename(filepath)
        save_name = os.path.splitext(filename)[0] + ".png"
        save_path = os.path.join(output_dir, save_name)
        
        print(f"[{i+1}/{len(files)}] Processing {filename}...")
        visualize_file(filepath, save_path=save_path)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize marmoset vocalizations')
    parser.add_argument('--data_dir', type=str, default='/home/hjalmar/Python/VocSeparator/test',
                       help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['clean', 'mixture', 'multiple', 'file', 'dir'], 
                       default='mixture', help='Visualization mode')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure (or output dir for dir mode)')
    parser.add_argument('--idx', type=int, default=0, help='Sample index')
    parser.add_argument('--file', type=str, default=None, help='Path to audio file for file mode')
    
    args = parser.parse_args()
    
    if args.mode == 'clean':
        visualize_clean_vocalization(args.data_dir, args.idx, args.save)
    elif args.mode == 'mixture':
        visualize_mixture(args.data_dir, args.idx, args.save)
    elif args.mode == 'multiple':
        visualize_multiple_mixtures(args.data_dir, 4, args.save)
    elif args.mode == 'file':
        if not args.file:
            print("Error: --file argument is required for file mode")
        else:
            visualize_file(args.file, args.save)
    elif args.mode == 'dir':
        # Use data_dir as input directory
        visualize_directory(args.data_dir, args.save)
        

def plot_training_history(history_path="training_history.pkl"):
    """
    Plots the training and validation loss curves from a pickle file.
    """
    if not os.path.exists(history_path):
        print(f"Error: History file '{history_path}' not found.")
        return

    with open(history_path, 'rb') as f:
        history = pickle.load(f)

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])

    if not train_loss:
        print("Error: No training loss data found in history.")
        return

    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (L1)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = "training_curves.png"
    plt.savefig(out_file)
    print(f"Saved training curves to {out_file}")
    
    # Print final stats
    print("\nTraining Summary:")
    print(f"Final Training Loss: {train_loss[-1]:.4f}")
    if val_loss:
        print(f"Final Validation Loss: {val_loss[-1]:.4f}")