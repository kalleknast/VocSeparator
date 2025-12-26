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
from visualize import save_sample_spectrograms


def train(data_dir, batch_size=32, num_epochs=100, learning_rate=1e-4):
    # Hyperparameters
    # data_dir argument passed from command line

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Datasets
    train_dataset = MarmosetDataset(data_dir, segment_length=16384, split='train')
    val_dataset = MarmosetDataset(data_dir, segment_length=16384, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Visualize training samples
    # save_sample_spectrograms(train_dataset, "images", num_samples=5)

    # Model
    model = WaveNetSourceSeparator(in_channels=1, out_channels=1).to(device)
    
    # Loss and Optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)

    print(f"Starting training on {device}...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Track history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
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
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(data_dir, "model_best.pth"))
            print(f"  -> Saved new best model: {best_val_loss:.6f}")        
        
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
