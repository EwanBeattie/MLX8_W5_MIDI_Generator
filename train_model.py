import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from asteroid.models import BaseModel
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from pathlib import Path
import random
import warnings
from tqdm import tqdm
import numpy as np

# Suppress the specific numpy tensor creation warning from Asteroid
warnings.filterwarnings("ignore", message="Creating a tensor from a list of numpy.ndarrays is extremely slow")

def load_model():
    """Load the customized ConvTasNet model for SATB separation."""
    print("Loading ConvTasNet model for training...")
    
    # Load base model structure
    model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
    
    # Modify for 4 sources (same as in models.py)
    old_output = model.masker.mask_net[1]
    
    # Update n_src parameter for 4 sources (SATB)
    model.masker.n_src = 4
    
    scale = 4 // 2
    new_out_channels = old_output.out_channels * scale

    model.masker.mask_net[1] = torch.nn.Conv1d(
        in_channels=old_output.in_channels,
        out_channels=new_out_channels,
        kernel_size=old_output.kernel_size[0],
        stride=old_output.stride[0],
        padding=old_output.padding[0],
        bias=old_output.bias is not None
    )
    
    # Initialize new layer weights
    torch.nn.init.xavier_uniform_(model.masker.mask_net[1].weight)
    
    # Load initial weights if available
    checkpoint_path = "checkpoints/convtasnet_satb_init.pth"
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded initial weights from {checkpoint_path}")
    
    return model

class SATBDataset(Dataset):
    """Dataset class for SATB audio chunks using NumPy arrays with memory mapping."""
    
    def __init__(self, indices, data_dir):
        self.indices = indices
        self.data_dir = Path(data_dir)
        
        # Load data with memory mapping for efficient access
        self.mixed_chunks = np.load(self.data_dir / "mixed_chunks.npy", mmap_mode='r')
        self.source_chunks = np.load(self.data_dir / "source_chunks.npy", mmap_mode='r')
        
        print(f"Loaded dataset with {len(indices)} chunks")
        print(f"  Mixed chunks shape: {self.mixed_chunks.shape}")
        print(f"  Source chunks shape: {self.source_chunks.shape}")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        chunk_idx = self.indices[idx]
        
        # Get data and convert to float tensors
        mixed = torch.from_numpy(self.mixed_chunks[chunk_idx].copy()).float() / 127.0
        sources = torch.from_numpy(self.source_chunks[chunk_idx].copy()).float() / 127.0
        
        return mixed, sources

def load_training_data():
    """Load training and validation data using NumPy arrays."""
    print("Loading training data...")
    
    data_dir = Path("processed_data_compressed")
    
    # Load indices for train/val splits
    train_indices = np.load(data_dir / "train_indices.npy")
    val_indices = np.load(data_dir / "val_indices.npy")
    
    print(f"Training chunks: {len(train_indices):,}")
    print(f"Validation chunks: {len(val_indices):,}")
    
    # Create datasets
    train_dataset = SATBDataset(train_indices, data_dir)
    val_dataset = SATBDataset(val_indices, data_dir)
    
    return train_dataset, val_dataset

def train_model():
    """Fine-tune the ConvTasNet model on SATB data."""
    print("🎵 Fine-tuning ConvTasNet for SATB Separation")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    model = load_model().to(device)
    train_dataset, val_dataset = load_training_data()
    
    # Training parameters
    learning_rate = 1e-4
    num_epochs = 10
    batch_size = 16  # Increased batch size for better GPU utilization
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # Use multiple workers for faster data loading
        pin_memory=True  # Speed up CPU->GPU transfer
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,  # Fewer workers for validation
        pin_memory=True
    )
    
    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    print(f"Training parameters:")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📈 Epoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        for mixed, sources in tqdm(train_loader, desc="Training", leave=False):
            mixed = mixed.to(device)
            sources = sources.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            separated = model(mixed)
            
            # Compute loss
            loss = loss_func(separated, sources)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for mixed, sources in tqdm(val_loader, desc="Validation", leave=False):
                mixed = mixed.to(device)
                sources = sources.to(device)
                
                separated = model(mixed)
                loss = loss_func(separated, sources)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = f"checkpoints/convtasnet_satb_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"   💾 Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_path = "checkpoints/convtasnet_satb_trained.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n✅ Training complete! Final model saved to: {final_path}")

if __name__ == "__main__":
    train_model()
