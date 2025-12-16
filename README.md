# VocSeparator

A WaveNet-based deep learning system for separating overlapping marmoset vocalizations from raw audio waveforms.

## Features

- **Energy-based onset/offset detection** to crop silence from recordings
- **Data augmentation** with random amplitude scaling and time shifting
- **Train/validation split** to prevent overfitting
- **WaveNet architecture** with dilated convolutions for raw waveform processing
- **Iterative separation** for multiple overlapping vocalizations
- **Visualization tools** for spectrograms and training curves

## Dataset

Current dataset: **38,065 clean vocalization files** (30,452 train, 7,613 validation)
- **Source**: [MarmAudio: A large annotated dataset of vocalizations by common marmosets](https://zenodo.org/records/15017207)
- Split: 80% train, 20% validation
- Files are split such that validation files are not used in training

### Data Augmentation

From 38,065 files, the augmentation creates:
- **~1.45 billion unique pairs per epoch** (30,452 × 30,451)
- **Effectively infinite variations** due to:
  - Random amplitude scaling (0.7-1.0)
  - Random time shifting (up to 25% of segment length)
  - Combined: ~12.5 million unique augmented versions per pair

## Installation

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch torchaudio numpy soundfile matplotlib
```

## Usage

### 1. Visualize Data

```bash
# Visualize a mixture and its components
python visualize.py --mode mixture --save mixture_example.png

# Visualize a clean vocalization
python visualize.py --mode clean --idx 0 --save clean_voc.png

# Visualize multiple mixtures
python visualize.py --mode multiple --save multiple_mixtures.png
```

### 2. Train the Model

```bash
# Train with default settings (100 epochs)
python train.py

# The script will:
# - Split data into train (72 files) and validation (18 files)
# - Print train and validation loss after each epoch
# - Save checkpoints every 10 epochs to checkpoints/
# - Save final model as model_final.pth
# - Save training history to training_history.pkl
# - Return history dict with train_loss and val_loss lists
```

### 3. Plot Training Curves

```bash
# After training, plot the loss curves
python plot_history.py --history training_history.pkl --save training_curves.png
```

### 4. Run Inference

```bash
# Separate a mixture into 2 sources
python inference.py --mixture test_mixture.wav --model model_final.pth --sources 2

# For 3 overlapping vocalizations
python inference.py --mixture test_mixture.wav --model model_final.pth --sources 3
```

### 5. Create Test Mixtures

```bash
# Generate a test mixture from the dataset
python create_mixture.py
# This creates test_mixture.wav
```

## File Structure

```
silver-voyager/
├── dataset.py           # Data loading, augmentation, train/val split
├── model.py             # WaveNet architecture
├── train.py             # Training loop with validation
├── inference.py         # Separation and iterative demuxing
├── visualize.py         # Spectrogram visualization
├── plot_history.py      # Plot training curves
├── create_mixture.py    # Helper to create test mixtures
├── venv/                # Virtual environment
├── checkpoints/         # Model checkpoints
├── model_final.pth      # Final trained model
└── training_history.pkl # Training/validation loss history
```

## Model Architecture

- **Input**: Raw waveform mixture (1 channel)
- **Output**: Separated vocalization (1 channel)
- **Layers**: 30 dilated convolutional residual blocks
- **Parameters**: ~500K
- **Receptive field**: Large (exponential dilation: 1, 2, 4, ..., 512)

## Training Configuration

- **Loss**: L1 Loss (MAE)
- **Optimizer**: Adam (lr=1e-4)
- **Batch size**: 8
- **Segment length**: 16,384 samples (~0.37s at 44.1kHz)
- **Epochs**: 100
- **Device**: CPU (GPU recommended for production)

## Data Recommendations

- **Current**: 90 files (minimal for testing)
- **Minimum**: 500-1,000 files for basic separation
- **Good**: 2,000-5,000 files for robust performance
- **Excellent**: 10,000+ files for state-of-the-art results

## Iterative Separation

For N overlapping vocalizations:
1. **Iteration 1**: `[voc1 + voc2 + voc3]` → model → `voc1`, remainder: `[voc2 + voc3]`
2. **Iteration 2**: `[voc2 + voc3]` → model → `voc2`, remainder: `voc3`
3. Continue until all sources separated

## Example Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Visualize some data
python visualize.py --mode mixture --save mixture_viz.png

# 3. Train the model
python train.py

# 4. Plot training progress
python plot_history.py

# 5. Create a test mixture
python create_mixture.py

# 6. Run inference
python inference.py --mixture test_mixture.wav --model model_final.pth
```

## Notes

- The model learns to predict `voc1` from `mixture = voc1 + voc2`
- Validation files are completely separate from training files
- Training history is automatically saved for later analysis
- Spectrograms help verify that mixtures are being created correctly
