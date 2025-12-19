import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import soundfile as sf
from dataset import MarmosetDataset

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
