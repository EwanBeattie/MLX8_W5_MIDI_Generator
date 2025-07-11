import torch
import torchaudio
from pathlib import Path

def load_and_process_voices():
    """Load, resample, convert to mono and normalize all voice files."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    target_sr = 8000
    
    songs = ["ER", "LI", "ND"]
    voices = ["soprano", "alto", "tenor", "bass"]
    max_singers = 4
    
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
    """Create chunks for first song, singer 1 from each voice, first 5 chunks only."""
    
    chunk_duration = 4.0  # seconds
    target_sr = 8000
    chunk_samples = int(chunk_duration * target_sr)
    
    # Use first song and singer 1 from each voice
    song = "ER"
    voices = ["soprano", "alto", "tenor", "bass"]
    singer_num = 1
    
    print(f"Creating chunks for {song} with singer {singer_num}")
    
    # Get the 4 voices for this combination
    voice_audios = []
    for voice in voices:
        if singer_num in audio_data[song][voice]:
            voice_audios.append(audio_data[song][voice][singer_num])
        else:
            print(f"  Warning: {song}_{voice}_{singer_num} not found")
            return []
    
    # Create mix
    mixed = sum(voice_audios)
    mixed = mixed / torch.max(torch.abs(mixed))
    
    # Create chunks (only first 5) - pre-stacked for fast access
    max_chunks = 5
    min_length = len(voice_audios[0])  # All are same length now
    
    mixed_chunks_list = []
    source_chunks_list = []
    
    for chunk_idx in range(max_chunks):
        start = chunk_idx * chunk_samples
        end = start + chunk_samples
        
        if end > min_length:
            break
            
        mixed_chunk = mixed[start:end]
        source_chunks = torch.stack([audio[start:end] for audio in voice_audios])
        
        mixed_chunks_list.append(mixed_chunk)
        source_chunks_list.append(source_chunks)
        
        print(f"  Chunk {chunk_idx}: {mixed_chunk.shape[0]} samples")
    
    # Stack into single tensors for fast indexing
    mixed_chunks = torch.stack(mixed_chunks_list)  # Shape: [5, 32000]
    source_chunks = torch.stack(source_chunks_list)  # Shape: [5, 4, 32000]
    
    return {
        'mixed': mixed_chunks,
        'sources': source_chunks,
        'count': len(mixed_chunks_list)
    }

if __name__ == "__main__":
    audio_data = load_and_process_voices()
    print("✅ All audio files loaded and processed")
    
    chunks = create_chunks(audio_data)
    print(f"✅ Created {chunks['count']} chunks")
    print(f"   Mixed chunks shape: {chunks['mixed'].shape}")
    print(f"   Source chunks shape: {chunks['sources'].shape}")


