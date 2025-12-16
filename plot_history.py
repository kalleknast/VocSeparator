import matplotlib.pyplot as plt
import pickle

def plot_training_history(history_path='training_history.pkl', save_path='training_curves.png'):
    """Plot training and validation loss curves from saved history."""
    
    # Load history
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (L1)', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary
    print(f"\nTraining Summary:")
    print(f"Final Training Loss: {history['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history['val_loss'][-1]:.4f}")
    print(f"Best Validation Loss: {min(history['val_loss']):.4f} (Epoch {history['val_loss'].index(min(history['val_loss'])) + 1})")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='training_history.pkl', 
                       help='Path to training history pickle file')
    parser.add_argument('--save', type=str, default='training_curves.png',
                       help='Path to save plot')
    
    args = parser.parse_args()
    plot_training_history(args.history, args.save)
