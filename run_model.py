import torch
import torchaudio
from asteroid.models import BaseModel
from pathlib import Path
import random

def load_model():
    """Load the customized ConvTasNet model for SATB separation."""
    print("Loading ConvTasNet model...")
    
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
    
    # Load trained weights
    checkpoint_path = "checkpoints/convtasnet_satb_init.pth"
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"✅ Loaded weights from {checkpoint_path}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_path}, using initialized weights")
    
    model.eval()
    return model

def load_test_data():
    """Load test data chunks."""
    print("Loading test data...")
    
    # Load test split filenames
    test_files = torch.load("processed_data_compressed/test_split.pt")
    chunks_dir = Path("processed_data_compressed/chunks")
    
    print(f"Found {len(test_files)} test chunks")
    return test_files, chunks_dir

def run_separation(model, mixed_audio):
    """Run SATB separation on mixed audio."""
    # Convert int8 back to float
    mixed_float = mixed_audio.float() / 127.0
    
    # Add batch dimension: [1, samples]
    mixed_batch = mixed_float.unsqueeze(0)
    
    with torch.no_grad():
        # Run separation
        separated = model(mixed_batch)
        # Remove batch dimension: [4, samples]
        separated = separated.squeeze(0)
    
    return separated

def main():
    """Main function to run SATB separation."""
    print("🎵 SATB Voice Separation with ConvTasNet")
    print("=" * 50)
    
    # Load model
    model = load_model()
    
    # Load test data
    test_files, chunks_dir = load_test_data()
    
    # Process a few random test samples
    num_samples = min(2, len(test_files))
    sample_files = random.sample(test_files, num_samples)
    
    voice_names = ["soprano", "alto", "tenor", "bass"]
    
    for i, filename in enumerate(sample_files):
        print(f"\n📊 Processing sample {i+1}/{num_samples}: {filename}")
        
        # Load chunk data
        chunk_data = torch.load(chunks_dir / filename)
        mixed = chunk_data['mixed']
        true_sources = chunk_data['sources']
        
        print(f"   Song: {chunk_data['song']}")
        print(f"   Singer combo: {chunk_data['singer_combo']}")
        
        # Run separation
        separated = run_separation(model, mixed)
        
        print(f"   Input shape: {mixed.shape}")
        print(f"   Output shape: {separated.shape}")
        print(f"   True sources shape: {true_sources.shape}")
        
        # Convert back to audio range for saving
        separated_audio = (separated * 127).clamp(-127, 127).to(torch.int8)
        
        # Save separated outputs (optional)
        output_dir = Path("separated_outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Convert to float for audio saving
        separated_float = separated.float() / 127.0
        mixed_float = mixed.float() / 127.0
        
        # Save mixed input
        torchaudio.save(
            output_dir / f"sample_{i+1}_mixed.wav",
            mixed_float.unsqueeze(0), 8000
        )
        
        # Save separated voices
        for j, voice_name in enumerate(voice_names):
            torchaudio.save(
                output_dir / f"sample_{i+1}_{voice_name}_separated.wav",
                separated_float[j].unsqueeze(0), 8000
            )
            
            # Also save ground truth for comparison
            true_float = true_sources[j].float() / 127.0
            torchaudio.save(
                output_dir / f"sample_{i+1}_{voice_name}_true.wav",
                true_float.unsqueeze(0), 8000
            )
        
        print(f"   ✅ Saved outputs to {output_dir}/")
    
    print(f"\n🎉 Separation complete!")
    print(f"   Check {output_dir}/ for audio outputs")
    print(f"   Compare '_separated' vs '_true' files to evaluate quality")

if __name__ == "__main__":
    main()
