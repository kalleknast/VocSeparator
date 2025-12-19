import os
import glob
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal
from tqdm import tqdm
import argparse

def measure_signal_to_noise(directory, high_pass_freq=4000, limit=None):
    """
    Measures signal statistics for audio files in the given directory.
    
    Metric:
    - Signal: Max amplitude above high_pass_freq (default 4000 Hz).
    - Noise: RMS amplitude below high_pass_freq.
    
    Returns:
    - pd.DataFrame containing filenames and metrics.
    """
    # Find all audio files (wav and flac)
    search_patterns = [
        os.path.join(directory, "**", "*.flac"),
        os.path.join(directory, "**", "*.wav")
    ]
    files = []
    for pattern in search_patterns:
        files.extend(glob.glob(pattern, recursive=True))
        
    if limit:
        import random
        random.shuffle(files)
        files = files[:limit]
        print(f"Limiting to {limit} random files.")

    results = []
    print(f"Found {len(files)} audio files (after limit) in {directory}. Starting analysis...")
    
    for filepath in tqdm(files, desc="Processing"):
        try:
            # Load audio using soundfile
            audio, sr = sf.read(filepath)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
                
            # Design filters (4th order Butterworth)
            # High-pass for Signal (Phee calls are high freq)
            sos_hp = scipy.signal.butter(4, high_pass_freq, 'hp', fs=sr, output='sos')
            signal_part = scipy.signal.sosfilt(sos_hp, audio)
            
            # Low-pass for Noise (Background is typically low freq)
            sos_lp = scipy.signal.butter(4, high_pass_freq, 'lp', fs=sr, output='sos')
            noise_part = scipy.signal.sosfilt(sos_lp, audio)
            
            # Compute metrics
            # "Signal": Max Amplitude above 4000 Hz
            max_amp_signal = np.max(np.abs(signal_part))
            
            # "Noise": RMS of low frequency part
            rms_noise = np.sqrt(np.mean(noise_part**2))
            
            # SNR Ratio (Signal / Noise)
            # Avoid divide by zero
            if rms_noise == 0:
                snr = np.inf 
            else:
                snr = max_amp_signal / rms_noise
            
            results.append({
                'filename': os.path.basename(filepath),
                'signal_max_amp_above_4khz': max_amp_signal,
                'noise_rms_below_4khz': rms_noise,
                'snr_estimate': snr,
                'filepath': filepath
            })
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Signal to Noise in Audio Files")
    parser.add_argument('--dir', type=str, default='Vocalizations', help='Directory to search for audio files')
    parser.add_argument('--out', type=str, default='snr_report.csv', help='Output CSV file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' not found.")
        exit(1)
        
    df = measure_signal_to_noise(args.dir, limit=args.limit)
    
    if not df.empty:
        # Sort by Signal strength ascending (lowest signal first)
        df_sorted = df.sort_values('signal_max_amp_above_4khz')
        
        print("\nAnalysis Complete.")
        print(f"Processed {len(df)} files.")
        print("\nTop 10 Files with WEAKEST Signal (Max Amp > 4kHz):")
        print(df_sorted[['filename', 'signal_max_amp_above_4khz', 'snr_estimate']].head(10))
        
        df.to_csv(args.out, index=False)
        print(f"\nFull results saved to {args.out}")
    else:
        print("No files processed.")
