import torch
import torchaudio
from pathlib import Path
import gc
from tqdm import tqdm
import random
import numpy as np


def preprocess_satb_data_efficient():
    """Memory-efficient SATB preprocessing for ConvTasNet fine-tuning with NumPy storage."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    output_dir = Path("processed_data_compressed")
    output_dir.mkdir(exist_ok=True)
    
    # Create samples directory for quality checking
    samples_dir = output_dir / "quality_samples"
    samples_dir.mkdir(exist_ok=True)
    samples_saved = False
    
    # Parameters
    target_sr = 8000  # Reduced from 16000 to 8000 for smaller files
    chunk_duration = 4.0  # seconds
    chunk_samples = int(chunk_duration * target_sr)
    
    songs = ["ER", "LI", "ND"]
    voices = ["soprano", "alto", "tenor", "bass"]
    
    # Use all 4 singers per voice for maximum combinations
    max_singers_to_use = 4  # This gives 256 combinations per song
    
    # Estimate total chunks (we'll resize arrays if needed)
    # Assume average audio length of ~3 minutes = 180 seconds at 8kHz = 1.44M samples
    # With 4-second chunks, that's ~45 chunks per combination
    estimated_chunks_per_combo = 45
    max_combinations = len(songs) * (max_singers_to_use ** 4)  # 3 * 256 = 768
    estimated_total_chunks = max_combinations * estimated_chunks_per_combo
    
    print(f"Estimated total chunks: {estimated_total_chunks:,}")
    print("Starting processing (will resize arrays if needed)...")
    
    # Pre-allocate NumPy arrays with estimated size
    mixed_chunks_list = []
    source_chunks_list = []
    
    # Process and collect chunks
    total_combinations = len(songs) * (max_singers_to_use ** 4)
    pbar = tqdm(total=total_combinations, desc="Processing combinations")
    
    for song in songs:
        for soprano_singer in range(1, max_singers_to_use + 1):
            for alto_singer in range(1, max_singers_to_use + 1):
                for tenor_singer in range(1, max_singers_to_use + 1):
                    for bass_singer in range(1, max_singers_to_use + 1):
                        
                        singer_combo = [soprano_singer, alto_singer, tenor_singer, bass_singer]
                        pbar.set_description(f"Song {song}: S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer}")
                        
                        # Load all 4 voices for this combination
                        voice_audios = []
                        skip_combo = False
                        
                        for i, voice in enumerate(voices):
                            singer_num = singer_combo[i]
                            audio_file = data_dir / f"CSD_{song}_{voice}_{singer_num}.wav"
                            
                            if not audio_file.exists():
                                skip_combo = True
                                break
                            
                            # Load and resample
                            waveform, orig_sr = torchaudio.load(audio_file)
                            if orig_sr != target_sr:
                                resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                                waveform = resampler(waveform)
                            
                            # Convert to mono and normalize
                            waveform = torch.mean(waveform, dim=0)
                            waveform = waveform / torch.max(torch.abs(waveform))
                            voice_audios.append(waveform)
                        
                        if skip_combo:
                            pbar.update(1)
                            continue
                        
                        # Find minimum length
                        min_length = min(len(audio) for audio in voice_audios)
                        voice_audios = [audio[:min_length] for audio in voice_audios]
                        
                        # Create mix
                        mixed = sum(voice_audios)
                        mixed = mixed / torch.max(torch.abs(mixed))
                        
                        # Process chunks and store in lists
                        chunk_count = 0
                        for start in range(0, min_length - chunk_samples + 1, chunk_samples):
                            end = start + chunk_samples
                            
                            mixed_chunk = mixed[start:end]
                            source_chunks_tensor = torch.stack([audio[start:end] for audio in voice_audios])
                            
                            # Convert to int8 for smaller storage
                            mixed_int8 = (mixed_chunk * 127).clamp(-127, 127).to(torch.int8)
                            sources_int8 = (source_chunks_tensor * 127).clamp(-127, 127).to(torch.int8)
                            
                            # Store in lists
                            mixed_chunks_list.append(mixed_int8.numpy())
                            source_chunks_list.append(sources_int8.numpy())
                            
                            # Save quality samples (skip first chunk as it's often blank)
                            if not samples_saved and chunk_count == 1:  # Use second chunk
                                print(f"  Saving quality samples for {song}...")
                                
                                # Convert back to float for audio saving
                                mixed_float = mixed_int8.float() / 127.0
                                sources_float = sources_int8.float() / 127.0
                                
                                # Save mixed audio
                                torchaudio.save(
                                    samples_dir / f"{song}_mixed.wav",
                                    mixed_float.unsqueeze(0), target_sr
                                )
                                
                                # Save individual voices
                                voice_names = ["soprano", "alto", "tenor", "bass"]
                                for i, voice_name in enumerate(voice_names):
                                    torchaudio.save(
                                        samples_dir / f"{song}_{voice_name}.wav",
                                        sources_float[i].unsqueeze(0), target_sr
                                    )
                                
                                samples_saved = True
                            
                            chunk_count += 1
                        
                        # Clear memory after each combination
                        del voice_audios, mixed
                        gc.collect()
                        pbar.update(1)
    
    pbar.close()
    
    # Convert lists to NumPy arrays
    print(f"Converting {len(mixed_chunks_list):,} chunks to NumPy arrays...")
    mixed_chunks = np.array(mixed_chunks_list, dtype=np.int8)
    source_chunks = np.array(source_chunks_list, dtype=np.int8)
    total_chunks = len(mixed_chunks_list)
    
    # Clear lists to free memory
    del mixed_chunks_list, source_chunks_list
    gc.collect()
    
    # Save NumPy arrays
    print("Saving NumPy arrays...")
    np.save(output_dir / "mixed_chunks.npy", mixed_chunks)
    np.save(output_dir / "source_chunks.npy", source_chunks)
    
    print(f"✅ Memory-efficient preprocessing complete!")
    print(f"   Mixed chunks saved to: {output_dir / 'mixed_chunks.npy'}")
    print(f"   Source chunks saved to: {output_dir / 'source_chunks.npy'}")
    print(f"   Quality samples saved to: {samples_dir}")
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   Sample rate: {target_sr} Hz")
    print(f"   Data type: int8 (compressed)")
    
    # Create train/test/validation splits
    print(f"\n📊 Creating data splits...")
    create_data_splits(total_chunks, output_dir)

def create_data_splits(total_chunks, output_dir):
    """Create train/test/validation splits using indices."""
    
    # Create indices for all chunks
    all_indices = list(range(total_chunks))
    
    # Shuffle with seed 42 for reproducibility
    random.seed(42)
    random.shuffle(all_indices)
    
    train_size = int(0.8 * total_chunks)
    val_size = int(0.1 * total_chunks)
    
    # Split the indices
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    # Save split indices as NumPy arrays
    np.save(output_dir / "train_indices.npy", np.array(train_indices))
    np.save(output_dir / "val_indices.npy", np.array(val_indices))
    np.save(output_dir / "test_indices.npy", np.array(test_indices))
    
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   Train indices: {len(train_indices):,} ({len(train_indices)/total_chunks*100:.1f}%)")
    print(f"   Validation indices: {len(val_indices):,} ({len(val_indices)/total_chunks*100:.1f}%)")
    print(f"   Test indices: {len(test_indices):,} ({len(test_indices)/total_chunks*100:.1f}%)")
    print(f"   Split files saved to: {output_dir}")

if __name__ == "__main__":
    preprocess_satb_data_efficient()
