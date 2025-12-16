import soundfile as sf
import torch
from dataset import MarmosetDataset
import os

def create_mixture():
    data_dir = "/home/hjalmar/Python/VocDemuxer/data"
    dataset = MarmosetDataset(data_dir, segment_length=16384)
    
    # Get a sample
    mixture, target = dataset[0]
    
    # Save mixture
    mixture_np = mixture.numpy().flatten()
    sf.write("test_mixture.wav", mixture_np, 44100)
    print("Created test_mixture.wav")

if __name__ == "__main__":
    create_mixture()
