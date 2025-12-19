import os
import glob
import random
import numpy as np
import torch
import torchaudio
import soundfile as sf
import scipy.signal
from torch.utils.data import Dataset

class MarmosetDataset(Dataset):
    def __init__(self, data_dir, sample_rate=44100, segment_length=16384, transform=None, 
                 split='train', val_split=0.2, random_seed=42, highpass_cutoff=1000):
        """
        Args:
            data_dir (str): Directory containing clean .wav files.
            sample_rate (int): Target sample rate.
            segment_length (int): Length of audio segments to return (in samples).
            transform (callable, optional): Optional transform to be applied on a sample.
            split (str): 'train' or 'val' to specify which split to use.
            val_split (float): Fraction of data to use for validation (default 0.2).
            random_seed (int): Random seed for reproducible splits.
        """
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.transform = transform
        self.split = split
        self.highpass_cutoff = highpass_cutoff
        
        # Find all wav and flac files
        wav_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)
        flac_files = glob.glob(os.path.join(data_dir, "**", "*.flac"), recursive=True)
        all_files = wav_files + flac_files
        if not all_files:
            print(f"Warning: No .wav or .flac files found in {data_dir}")
            
        # Sort for reproducibility
        all_files = sorted(all_files)
        
        # Split into train and validation
        random.seed(random_seed)
        indices = list(range(len(all_files)))
        random.shuffle(indices)
        
        val_size = int(len(all_files) * val_split)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
        
        if split == 'train':
            self.file_list = [all_files[i] for i in train_indices]
        elif split == 'val':
            self.file_list = [all_files[i] for i in val_indices]
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split}")
            
        print(f"{split.upper()} split: {len(self.file_list)} files")
            
    def __len__(self):
        # We can define length arbitrarily since we are pairing randomly, 
        # but let's say it's equal to the number of files for now.
        return len(self.file_list)

    def load_and_crop(self, filepath, highpass_cutoff=1000):
        # waveform, sr = torchaudio.load(filepath)
        # Use soundfile directly to avoid torchaudio backend issues
        audio, sr = sf.read(filepath)
        
        # Apply Zero-Phase Highpass Filter
        if highpass_cutoff is not None and highpass_cutoff > 0:
            # Design filter (4th order Butterworth)
            b, a = scipy.signal.butter(4, highpass_cutoff, btype='highpass', fs=sr)
            # Apply forward-backward filter to ensure zero phase shift
            # Use axis=0 for time dimension (sf.read returns (time, channels) or (time,))
            padlen = min(len(audio)-1, 3 * max(len(a), len(b)))
            # If signal is too short for default padlen, reduce it or skip
            try:
                audio = scipy.signal.filtfilt(b, a, audio, axis=0, padlen=padlen).copy()
            except Exception as e:
                print(f"Warning: Could not filter {filepath}: {e}")

        waveform = torch.from_numpy(audio).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t() # (Time, Channels) -> (Channels, Time)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Energy-based activity detection
        # Calculate energy in windows
        frame_length = 1024
        hop_length = 512
        if waveform.shape[1] < frame_length:
            # File is too short for energy windowing; treat whole file as active
            pass
        else:
            energy = torch.norm(waveform.unfold(1, frame_length, hop_length), dim=2)
        
            # Simple thresholding (e.g., 10% of max energy)
            threshold = 0.1 * torch.max(energy)
            active_frames = torch.where(energy > threshold)[1]
            
            if len(active_frames) > 0:
                start_frame = max(0, active_frames[0] - 1)
                end_frame = min(energy.shape[1] - 1, active_frames[-1] + 1)
                
                start_sample = start_frame * hop_length
                end_sample = (end_frame + 1) * hop_length + frame_length
                
                # Clamp to valid range
                start_sample = max(0, start_sample)
                end_sample = min(waveform.shape[1], end_sample)
                
                waveform = waveform[:, start_sample:end_sample]
        
        return waveform

    def __getitem__(self, idx):
        # Pick two distinct files
        file1 = self.file_list[idx]
        file2 = random.choice(self.file_list)
        while file1 == file2 and len(self.file_list) > 1:
            file2 = random.choice(self.file_list)
            
        voc1 = self.load_and_crop(file1, highpass_cutoff=self.highpass_cutoff)
        voc2 = self.load_and_crop(file2, highpass_cutoff=self.highpass_cutoff)
        
        # Pad or crop to segment_length
        def adjust_length(wav, length):
            if wav.shape[1] < length:
                padding = length - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, padding))
            elif wav.shape[1] > length:
                start = random.randint(0, wav.shape[1] - length)
                wav = wav[:, start:start+length]
            return wav
            
        voc1 = adjust_length(voc1, self.segment_length)
        voc2 = adjust_length(voc2, self.segment_length)
        
        # Data augmentation: Random amplitude scaling
        # Scale each vocalization independently (0.7 to 1.0)
        scale1 = random.uniform(0.7, 1.0)
        scale2 = random.uniform(0.7, 1.0)
        voc1_scaled = voc1 * scale1
        voc2_scaled = voc2 * scale2
        
        # Data augmentation: Time shifting (onset offset)
        # Create a buffer with zeros to allow time shifts
        max_shift = self.segment_length // 4  # Allow up to 25% shift
        
        # Random time shifts for each vocalization
        shift1 = random.randint(0, max_shift)
        shift2 = random.randint(0, max_shift)
        
        # Create shifted versions by padding and cropping
        voc1_shifted = torch.nn.functional.pad(voc1_scaled, (shift1, 0))[:, :self.segment_length]
        voc2_shifted = torch.nn.functional.pad(voc2_scaled, (shift2, 0))[:, :self.segment_length]
        
        # Create mixture with augmented vocalizations
        mixture = voc1_shifted + voc2_shifted
        
        # Target is the first vocalization (shifted and scaled)
        target = voc1_shifted
        
        # Normalize mixture to avoid clipping and scale inputs
        max_val = torch.max(torch.abs(mixture))
        if max_val > 0:
            mixture = mixture / max_val
            voc1 = voc1 / max_val
            voc2 = voc2 / max_val # Keep consistent scaling
            
        return mixture, voc1


