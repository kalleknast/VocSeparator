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
    """Visualize a single clean vocalization."""
    dataset = MarmosetDataset(data_dir, split='train')
    
    # Load a single file directly
    filepath = dataset.file_list[file_idx]
    waveform = dataset.load_and_crop(filepath)
    
    fig = plot_spectrogram(waveform, dataset.sample_rate, 
                          title=f"Clean Vocalization: {filepath.split('/')[-1]}")
    
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

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize marmoset vocalizations')
    parser.add_argument('--data_dir', type=str, default='/home/hjalmar/Python/VocDemuxer/data',
                       help='Path to data directory')
    parser.add_argument('--mode', type=str, choices=['clean', 'mixture', 'multiple'], 
                       default='mixture', help='Visualization mode')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    parser.add_argument('--idx', type=int, default=0, help='Sample index')
    
    args = parser.parse_args()
    
    if args.mode == 'clean':
        visualize_clean_vocalization(args.data_dir, args.idx, args.save)
    elif args.mode == 'mixture':
        visualize_mixture(args.data_dir, args.idx, args.save)
    elif args.mode == 'multiple':
        visualize_multiple_mixtures(args.data_dir, 4, args.save)
