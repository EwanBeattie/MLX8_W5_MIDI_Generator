import torch
from asteroid.models import BaseModel
import os
from pathlib import Path

def get_satb_model():
    """Load and configure ConvTasNet model for SATB separation."""
    # Load the bare ConvTasNet model
    model = BaseModel.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")

    # Confirm current output layer
    old_output = model.masker.mask_net[1]  # The final Conv1d layer
    print(f"Original output layer: {old_output}")

    # Update n_src parameter for 4 sources (SATB) - MUST be set before layer modification
    model.masker.n_src = 4

    # Calculate new output channels for 4 sources
    # It was originally 2 × 512 → now 4 × 512
    scale = 4 // 2
    new_out_channels = old_output.out_channels * scale

    # Replace the output layer
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
    
    return model

# Run when script is executed directly
if __name__ == "__main__":
    model = get_satb_model()
    
    # Save modified model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/convtasnet_satb_init.pth")
    
    print("✅ Model updated to output 4 sources and saved to checkpoints/convtasnet_satb_init.pth")
