import torch
import torchaudio
from pathlib import Path
import gc
from tqdm import tqdm

def preprocess_satb_data_efficient():
    """Memory-efficient SATB preprocessing for ConvTasNet fine-tuning."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    target_sr = 16000
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
                            
                            # Save individual chunk files instead of keeping in memory
                            chunk_filename = f"{song}_S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer}_chunk{chunk_count}.pt"
                            chunk_data = {
                                'mixed': mixed_chunk.half(),  # Use half precision to save memory
                                'sources': source_chunks.half(),
                                'song': song,
                                'singer_combo': singer_combo,
                                'chunk_id': chunk_count
                            }
                            
                            torch.save(chunk_data, output_dir / chunk_filename)
                            chunk_count += 1
                        
                        # Clear memory after each combination
                        del voice_audios, mixed
                        gc.collect()
                        pbar.update(1)
    
    pbar.close()
    print(f"✅ Memory-efficient preprocessing complete!")
    print(f"   Individual chunk files saved to: {output_dir}")

if __name__ == "__main__":
    preprocess_satb_data_efficient()
