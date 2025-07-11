from models import get_satb_model
from data import get_data_loaders
import torch
from asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from tqdm import tqdm

def train():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model and data
    model = get_satb_model().to(device)
    train_loader, val_loader, test_loader = get_data_loaders(batch_size=4)
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")
    
    # Simple training loop
    model.train()
    for epoch in range(3):
        print(f"\nEpoch {epoch + 1}")
        
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (mixed, sources) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}")):
            mixed = mixed.to(device)
            sources = sources.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            separated = model(mixed)
            loss = loss_func(separated, sources)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        print(f"  Average Loss: {avg_loss:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    train()