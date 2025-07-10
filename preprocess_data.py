import torch
import torchaudio
from pathlib import Path
import gc
from tqdm import tqdm
import random


def preprocess_satb_data_efficient():
    """Memory-efficient SATB preprocessing for ConvTasNet fine-tuning."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    output_dir = Path("processed_data_compressed")
    output_dir.mkdir(exist_ok=True)
    
    # Create chunks subdirectory
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    
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
    
    total_combinations = len(songs) * max_singers_to_use * max_singers_to_use * max_singers_to_use * max_singers_to_use
    pbar = tqdm(total=total_combinations, desc="Processing combinations")
    
    for song in songs:
        print(f"Processing song: {song}")
        
        for soprano_singer in range(1, max_singers_to_use + 1):
            for alto_singer in range(1, max_singers_to_use + 1):
                for tenor_singer in range(1, max_singers_to_use + 1):
                    for bass_singer in range(1, max_singers_to_use + 1):
                        
                        singer_combo = [soprano_singer, alto_singer, tenor_singer, bass_singer]
                        pbar.set_description(f"Song {song}: S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer}")
                        
                        # Load all 4 voices for this combination
                        voice_audios = []
                        for i, voice in enumerate(voices):
                            singer_num = singer_combo[i]
                            audio_file = data_dir / f"CSD_{song}_{voice}_{singer_num}.wav"
                            
                            if not audio_file.exists():
                                print(f"Warning: {audio_file} not found")
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
                        
                        if len(voice_audios) != 4:
                            pbar.update(1)
                            continue
                            
                        # Find minimum length
                        min_length = min(len(audio) for audio in voice_audios)
                        voice_audios = [audio[:min_length] for audio in voice_audios]
                        
                        # Create mix
                        mixed = sum(voice_audios)
                        mixed = mixed / torch.max(torch.abs(mixed))
                        
                        # Process chunks immediately and save as smaller files
                        chunk_count = 0
                        for start in range(0, min_length - chunk_samples + 1, chunk_samples):
                            end = start + chunk_samples
                            
                            mixed_chunk = mixed[start:end]
                            source_chunks = torch.stack([audio[start:end] for audio in voice_audios])
                            
                            # Convert to int8 for smaller storage (from float16)
                            mixed_int8 = (mixed_chunk * 127).clamp(-127, 127).to(torch.int8)
                            sources_int8 = (source_chunks * 127).clamp(-127, 127).to(torch.int8)
                            
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
                            
                            # Save individual chunk files instead of keeping in memory
                            chunk_filename = f"{song}_S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer}_chunk{chunk_count}.pt"
                            chunk_data = {
                                'mixed': mixed_int8,
                                'sources': sources_int8,
                                'song': song,
                                'singer_combo': singer_combo,
                                'chunk_id': chunk_count
                            }
                            
                            torch.save(chunk_data, chunks_dir / chunk_filename)
                            chunk_count += 1
                        
                        # Clear memory after each combination
                        del voice_audios, mixed
                        gc.collect()
                        pbar.update(1)
    
    pbar.close()
    
    print(f"✅ Memory-efficient preprocessing complete!")
    print(f"   Individual chunk files saved to: {chunks_dir}")
    print(f"   Quality samples saved to: {samples_dir}")
    print(f"   Sample rate: {target_sr} Hz")
    print(f"   Data type: int8 (compressed from float16)")
    
    # Create train/test/validation splits
    print(f"\n📊 Creating data splits...")
    create_data_splits(chunks_dir, output_dir)

def create_data_splits(chunks_dir, output_dir):
    """Create train/test/validation splits from individual chunk files."""
    
    # Get all chunk files from chunks subdirectory
    all_chunk_files = list(chunks_dir.glob("*.pt"))
    all_chunk_files = [f.name for f in all_chunk_files]  # Just the filenames
    
    # Shuffle with seed 42 for reproducibility
    random.seed(42)
    random.shuffle(all_chunk_files)
    
    total_chunks = len(all_chunk_files)
    train_size = int(0.8 * total_chunks)
    val_size = int(0.1 * total_chunks)
    
    # Split the filenames
    train_files = all_chunk_files[:train_size]
    val_files = all_chunk_files[train_size:train_size + val_size]
    test_files = all_chunk_files[train_size + val_size:]
    
    # Save split files to main output directory (not chunks subdirectory)
    torch.save(train_files, output_dir / "train_split.pt")
    torch.save(val_files, output_dir / "val_split.pt") 
    torch.save(test_files, output_dir / "test_split.pt")
    
    print(f"   Total chunks: {total_chunks:,}")
    print(f"   Train files: {len(train_files):,} ({len(train_files)/total_chunks*100:.1f}%)")
    print(f"   Validation files: {len(val_files):,} ({len(val_files)/total_chunks*100:.1f}%)")
    print(f"   Test files: {len(test_files):,} ({len(test_files)/total_chunks*100:.1f}%)")
    print(f"   Split files saved to: {output_dir}")

if __name__ == "__main__":
    preprocess_satb_data_efficient()
