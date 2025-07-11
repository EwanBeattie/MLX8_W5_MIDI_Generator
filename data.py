from torch.utils.data import Dataset, DataLoader
import pickle

class SATBDataset(Dataset):
    """Simple SATB Dataset for loading chunks."""
    
    def __init__(self, chunks_data, indices):
        self.mixed_chunks = chunks_data['mixed']
        self.source_chunks = chunks_data['sources']
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        chunk_idx = self.indices[idx]
        mixed = self.mixed_chunks[chunk_idx]
        sources = self.source_chunks[chunk_idx]
        return mixed, sources

def get_data_loaders(chunks_file="chunks.pkl", batch_size=8):
    """Load chunks and create train/val DataLoaders."""
    
    # Load chunks
    with open(chunks_file, 'rb') as f:
        chunks_data = pickle.load(f)
    
    total_chunks = chunks_data['count']
    
    # Simple 80/10/10 split
    train_size = int(0.8 * total_chunks)
    val_size = int(0.1 * total_chunks)
    
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_chunks))
    
    # Create datasets
    train_dataset = SATBDataset(chunks_data, train_indices)
    val_dataset = SATBDataset(chunks_data, val_indices)
    test_dataset = SATBDataset(chunks_data, test_indices)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples") 
    print(f"Test: {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader