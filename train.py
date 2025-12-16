import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MarmosetDataset
from model import WaveNetSourceSeparator

def train():
    # Hyperparameters
    data_dir = "/home/hjalmar/Python/VocDemuxer/data"
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
    train()
