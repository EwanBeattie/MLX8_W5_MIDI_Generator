#!/usr/bin/env python3
"""
Upload processed SATB audio data to Hugging Face Hub.
"""

from huggingface_hub import HfApi, login, create_repo
from pathlib import Path
import os
import glob
from tqdm import tqdm

def upload_processed_data():
    """Upload processed audio data to Hugging Face."""
    
    # Get username automatically
    api = HfApi()
    try:
        user_info = api.whoami()
        username = user_info['name']
        repo_name = f"{username}/satb-choral-dataset"
    except:
        print("Please enter your Hugging Face username:")
        username = input("Username: ").strip()
        repo_name = f"{username}/satb-choral-dataset"
    
    processed_data_dir = Path("processed_data_compressed")
    
    print(f"🚀 Uploading SATB Choral Dataset to: {repo_name}")
    
    # Step 1: Login (you'll need to provide your token)
    print("\n1. Logging in to Hugging Face...")
    try:
        # Try to use existing token first
        user = api.whoami()
        print(f"   Already logged in as: {user['name']}")
    except:
        print("   Please login with your Hugging Face token:")
        login()
        # Get user info after login
        user = api.whoami()
        username = user['name']
        repo_name = f"{username}/satb-choral-dataset"
    
    # Step 2: Create repository
    print(f"\n2. Creating repository: {repo_name}")
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            exist_ok=True,
            private=False  # Set to True if you want a private dataset
        )
        print(f"   ✅ Repository created/exists: https://huggingface.co/datasets/{repo_name}")
    except Exception as e:
        print(f"   ⚠️  Repository creation failed: {e}")
        return
    
    # Step 3: Create dataset card (README.md)
    print("\n3. Creating dataset card...")
    create_dataset_card(processed_data_dir)
    
    # Step 4: Upload files
    print("\n4. Uploading files...")
    
    # Get list of all files to upload
    all_files = list(processed_data_dir.glob("*.pt"))
    readme_file = processed_data_dir / "README.md"
    
    if readme_file.exists():
        all_files.append(readme_file)
    
    print(f"   Found {len(all_files)} files to upload")
    print("   This will take a while... Please be patient.")
    
    # Use upload_large_folder for better handling of large datasets
    try:
        print("   Using upload_large_folder for better handling of large datasets...")
        api.upload_large_folder(
            folder_path=str(processed_data_dir),
            repo_id=repo_name,
            repo_type="dataset",
            ignore_patterns=["*.tmp", "*.log"]  # Ignore temporary files
        )
        print("   ✅ All files uploaded successfully!")
    except Exception as e:
        print(f"   ⚠️ Large folder upload failed: {e}")
        print("   Falling back to batch uploads...")
        
        # Fallback to batch uploads - upload in smaller groups
        print("   Uploading in batches for better reliability...")
        
        # Upload README first
        if readme_file.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(readme_file),
                    path_in_repo="README.md",
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print("   ✅ README uploaded")
            except Exception as e:
                print(f"   ❌ Failed to upload README: {e}")
        
        # Upload split files
        split_files = list(processed_data_dir.glob("*_split.pt"))
        for split_file in split_files:
            try:
                api.upload_file(
                    path_or_fileobj=str(split_file),
                    path_in_repo=split_file.name,
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print(f"   ✅ {split_file.name} uploaded")
            except Exception as e:
                print(f"   ❌ Failed to upload {split_file.name}: {e}")
        
        # Upload quality samples folder
        samples_dir = processed_data_dir / "quality_samples"
        if samples_dir.exists():
            try:
                api.upload_folder(
                    folder_path=str(samples_dir),
                    path_in_repo="quality_samples",
                    repo_id=repo_name,
                    repo_type="dataset"
                )
                print("   ✅ Quality samples uploaded")
            except Exception as e:
                print(f"   ❌ Failed to upload quality samples: {e}")
        
        # Upload chunks in batches of 2000 files
        chunks_dir = processed_data_dir / "chunks"
        if chunks_dir.exists():
            chunk_files = list(chunks_dir.glob("*.pt"))
            batch_size = 2000
            
            for i in range(0, len(chunk_files), batch_size):
                batch = chunk_files[i:i + batch_size]
                print(f"   Uploading chunk batch {i//batch_size + 1}/{(len(chunk_files) + batch_size - 1)//batch_size} ({len(batch)} files)...")
                
                for j, chunk_file in enumerate(batch):
                    try:
                        api.upload_file(
                            path_or_fileobj=str(chunk_file),
                            path_in_repo=f"chunks/{chunk_file.name}",
                            repo_id=repo_name,
                            repo_type="dataset"
                        )
                        if j % 500 == 0:  # Progress every 500 files within batch
                            print(f"     Progress: {j+1}/{len(batch)} files in current batch")
                    except Exception as e:
                        print(f"     ❌ Failed to upload {chunk_file.name}: {e}")
                
                print(f"   ✅ Batch {i//batch_size + 1} completed")
    
    print(f"\n✅ Upload complete!")
    print(f"   Dataset URL: https://huggingface.co/datasets/{repo_name}")
    print(f"   You can now use this dataset with: datasets.load_dataset('{repo_name}')")

def create_dataset_card(output_dir):
    """Create a README.md file describing the dataset."""
    
    # Count files and estimate size (including chunks subdirectory)
    pt_files = list(output_dir.glob("**/*.pt"))
    wav_files = list(output_dir.glob("**/*.wav"))
    total_size_mb = sum(f.stat().st_size for f in pt_files + wav_files) / (1024 * 1024)
    
    readme_content = f"""---
license: cc-by-4.0
task_categories:
- audio-classification
- audio-to-audio
tags:
- audio
- music
- choral
- satb
- source-separation
- pytorch
pretty_name: SATB Choral Source Separation Dataset (Compressed)
size_categories:
- 1K<n<10K
---

# SATB Choral Source Separation Dataset (Compressed)

This dataset contains preprocessed 4-second audio chunks from the Choral Singing Dataset (CSD) formatted for SATB (Soprano, Alto, Tenor, Bass) voice source separation tasks.

**This is the compressed version with 8kHz sample rate and int8 precision for smaller file sizes.**

## Dataset Description

- **Total chunks**: {len(pt_files):,}
- **Dataset size**: ~{total_size_mb:.1f} MB
- **Audio format**: 8 kHz mono, 4-second chunks  
- **Precision**: int8 for memory efficiency
- **Sources**: 4 individual voice parts (SATB)
- **Mixed audio**: Sum of all 4 voices

## Data Structure

### Folder Structure
```
├── chunks/                    # All individual chunk .pt files
├── quality_samples/           # Sample WAV files for quality checking
├── train_split.pt            # List of training chunk filenames
├── val_split.pt              # List of validation chunk filenames
└── test_split.pt             # List of test chunk filenames
```

### Chunk Files
Each `.pt` file in `chunks/` contains a dictionary with:
- `mixed`: Mixed audio tensor (4 seconds, 32k samples at 8kHz) - int8
- `sources`: Individual voice tensors [4, 32k] (soprano, alto, tenor, bass) - int8
- `song`: Song identifier ("ER", "LI", or "ND")
- `singer_combo`: List of singer IDs used [soprano_id, alto_id, tenor_id, bass_id]
- `chunk_id`: Chunk number within the combination

## Usage

```python
import torch

# Load data splits
train_files = torch.load('train_split.pt')
val_files = torch.load('val_split.pt')
test_files = torch.load('test_split.pt')

# Load a single chunk
chunk = torch.load(f'chunks/{{train_files[0]}}')
mixed_audio = chunk['mixed'].float() / 127.0  # Convert back to float
separated_voices = chunk['sources'].float() / 127.0

# mixed_audio shape: [32000] (4 seconds at 8kHz)
# separated_voices shape: [4, 32000] (SATB voices)
```

## Compression Details

- **Sample rate**: Reduced from 16kHz to 8kHz (2x reduction)
- **Precision**: int8 instead of float16 (2x reduction)  
- **Total compression**: ~4x smaller than original
- **Quality samples**: Available in `quality_samples/` for evaluation

## Original Dataset

Based on the Choral Singing Dataset (CSD):
- **Songs**: "Erbarme dich" (ER), "Locus iste" (LI), "Nuper rosarum flores" (ND)
- **Singers**: 4 different singers per voice part
- **Combinations**: All possible singer combinations (4^4 = 256 per song)

## Citation

If you use this dataset, please cite the original CSD paper and this preprocessing work.

## License

CC-BY-4.0 - Same as the original Choral Singing Dataset
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"   Created dataset card: {readme_path}")

if __name__ == "__main__":
    # Check if processed data exists
    if not Path("processed_data_compressed").exists():
        print("❌ processed_data_compressed directory not found!")
        print("   Please run preprocess_data.py first")
        exit(1)
    
    # Check if there are .pt files
    pt_files = list(Path("processed_data_compressed").glob("**/*.pt"))
    if not pt_files:
        print("❌ No .pt files found in processed_data_compressed directory!")
        print("   Please run preprocess_data.py first")
        exit(1)
    
    print(f"Found {len(pt_files)} processed files")
    upload_processed_data()
