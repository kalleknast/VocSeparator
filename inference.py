import torch
import soundfile as sf
import argparse
import numpy as np
from model import WaveNetSourceSeparator

def separate(mixture_path, model_path, output_dir, num_sources=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = WaveNetSourceSeparator(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load audio
    audio, sr = sf.read(mixture_path)
    mixture = torch.from_numpy(audio).float()
    if mixture.ndim == 1:
        mixture = mixture.unsqueeze(0) # (1, Time)
    else:
        mixture = mixture.t()
        
    mixture = mixture.unsqueeze(0).to(device) # (1, 1, Time)
    
    current_mixture = mixture
    separated_sources = []
    
    with torch.no_grad():
        for i in range(num_sources - 1):
            # Predict one source
            estimated_source = model(current_mixture)
            separated_sources.append(estimated_source.squeeze().cpu().numpy())
            
            # Subtract from mixture to get the rest
            # Note: The model might output a scaled version, but we trained with L1/MSE 
            # so it should match amplitude if the mixture was simple addition.
            current_mixture = current_mixture - estimated_source
            
        # The remainder is the last source
        separated_sources.append(current_mixture.squeeze().cpu().numpy())
        
    # Save results
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(mixture_path))[0]
    
    for i, source in enumerate(separated_sources):
        out_path = os.path.join(output_dir, f"{base_name}_source_{i+1}.wav")
        sf.write(out_path, source, sr)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixture", type=str, required=True, help="Path to mixture wav file")
    parser.add_argument("--model", type=str, default="model_final.pth", help="Path to trained model")
    parser.add_argument("--out", type=str, default="output", help="Output directory")
    parser.add_argument("--sources", type=int, default=2, help="Number of sources to separate")
    
    args = parser.parse_args()
    
    separate(args.mixture, args.model, args.out, args.sources)
