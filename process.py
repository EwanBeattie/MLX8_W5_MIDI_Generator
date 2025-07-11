import torch
import torchaudio
from pathlib import Path
import pickle
from tqdm import tqdm

def load_and_process_voices():
    """Load, resample, convert to mono and normalize all voice files."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    target_sr = 8000
    
    songs = ["ER", "LI", "ND"]
    voices = ["soprano", "alto", "tenor", "bass"]
    max_singers = 2
    
    # Dictionary to store all processed audio
    processed_audio = {}
    
    for song in songs:
        processed_audio[song] = {}
        print(f"Processing song: {song}")
        
        # First pass: load all audio for this song
        song_audios = {}
        all_lengths = []
        
        for voice in voices:
            song_audios[voice] = {}
            
            for singer_num in range(1, max_singers + 1):
                audio_file = data_dir / f"CSD_{song}_{voice}_{singer_num}.wav"
                
                if not audio_file.exists():
                    print(f"  Warning: {audio_file} not found")
                    continue
                
                # Load and resample
                waveform, orig_sr = torchaudio.load(audio_file)
                if orig_sr != target_sr:
                    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
                    waveform = resampler(waveform)
                
                # Convert to mono and normalize
                waveform = torch.mean(waveform, dim=0)
                waveform = waveform / torch.max(torch.abs(waveform))
                
                song_audios[voice][singer_num] = waveform
                all_lengths.append(len(waveform))
        
        # Find minimum length for this song
        if all_lengths:
            min_length = min(all_lengths)
            print(f"  Min length for {song}: {min_length} samples")
            
            # Second pass: truncate all to same length and store
            processed_audio[song] = {}
            for voice in voices:
                processed_audio[song][voice] = {}
                for singer_num in song_audios[voice]:
                    truncated = song_audios[voice][singer_num][:min_length]
                    processed_audio[song][voice][singer_num] = truncated
                    print(f"  Loaded {song}_{voice}_{singer_num}: {len(truncated)} samples")
    
    return processed_audio

def create_chunks(audio_data):
    """Create chunks for all songs with all combinations of singers."""
    
    chunk_duration = 4.0  # seconds
    target_sr = 8000
    chunk_samples = int(chunk_duration * target_sr)
    
    songs = ["ER", "LI", "ND"]
    voices = ["soprano", "alto", "tenor", "bass"]
    max_singers = 2  # Reduced from 4 to 2 - uses only first 2 singers per voice
    
    all_mixed_chunks = []
    all_source_chunks = []
    
    total_combinations = 0
    
    # Create progress bar for all combinations
    total_expected_combinations = max_singers ** 4 * len(songs)
    pbar = tqdm(total=total_expected_combinations, desc="Processing combinations")
    
    for soprano_singer in range(1, max_singers + 1):
        for alto_singer in range(1, max_singers + 1):
            for tenor_singer in range(1, max_singers + 1):
                for bass_singer in range(1, max_singers + 1):
                    
                    singer_combo = [soprano_singer, alto_singer, tenor_singer, bass_singer]
                    
                    for song in songs:
                        total_combinations += 1
                        pbar.set_description(f"S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer} - {song}")
                        
                        # Get the 4 voices for this combination
                        voice_audios = []
                        skip_combo = False
                        
                        for i, voice in enumerate(voices):
                            singer_num = singer_combo[i]
                            if singer_num in audio_data[song][voice]:
                                voice_audios.append(audio_data[song][voice][singer_num])
                            else:
                                skip_combo = True
                                break
                        
                        if skip_combo:
                            pbar.update(1)
                            continue
                        
                        # Create mix
                        mixed = sum(voice_audios)
                        mixed = mixed / torch.max(torch.abs(mixed))
                        
                        # Create all possible chunks for this combination
                        min_length = len(voice_audios[0])  # All are same length now
                        num_chunks = (min_length - chunk_samples) // chunk_samples + 1
                        
                        for chunk_idx in range(num_chunks):
                            start = chunk_idx * chunk_samples
                            end = start + chunk_samples
                            
                            if end > min_length:
                                break
                                
                            mixed_chunk = mixed[start:end]
                            source_chunks = torch.stack([audio[start:end] for audio in voice_audios])
                            
                            all_mixed_chunks.append(mixed_chunk)
                            all_source_chunks.append(source_chunks)
                        
                        pbar.update(1)
    
    pbar.close()
    
    print(f"\n📊 Total singer combinations processed: {total_combinations}")
    print(f"📊 Expected: {max_singers**4} combinations")
    
    # Stack into single tensors for fast indexing
    mixed_chunks = torch.stack(all_mixed_chunks)  # Shape: [total_chunks, 32000]
    source_chunks = torch.stack(all_source_chunks)  # Shape: [total_chunks, 4, 32000]
    
    print(f"\n✅ Total chunks created: {len(all_mixed_chunks)}")
    
    chunks_data = {
        'mixed': mixed_chunks,
        'sources': source_chunks,
        'count': len(all_mixed_chunks)
    }
    
    chunks_file = "chunks.pkl"
    with open(chunks_file, 'wb') as f:
        pickle.dump(chunks_data, f)
    
    print(f"💾 Saved chunks to: {chunks_file}")
    
    return chunks_data

if __name__ == "__main__":
    audio_data = load_and_process_voices()
    print("✅ All audio files loaded and processed")
    
    chunks = create_chunks(audio_data)
    print(f"✅ Created {chunks['count']} chunks")
    print(f"   Mixed chunks shape: {chunks['mixed'].shape}")
    print(f"   Source chunks shape: {chunks['sources'].shape}")


