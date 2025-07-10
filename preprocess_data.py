import torch
import torchaudio
from pathlib import Path
import os

def save_audio_samples():
    """Save some mixed audio samples as WAV files for listening."""
    
    # output_dir = Path("processed_data")
    # samples_dir = output_dir / "audio_samples"
    # samples_dir.mkdir(exist_ok=True)
    
    # # Load processed data
    # train_data = torch.load(output_dir / "train_data.pt")
    # val_data = torch.load(output_dir / "val_data.pt")
    
    # print("Saving audio samples for listening...")
    
    # # Save 5th chunk from first song only
    # saved_songs = set()
    # sample_count = 0
    
    # for data_list, split_name in [(train_data, "train"), (val_data, "val")]:
    #     for i, chunk in enumerate(data_list):
    #         song = chunk['song']
            
    #         # Only save 5th chunk (index 4) from first song
    #         if song not in saved_songs and i == 4 and sample_count < 1:
    #             mixed = chunk['mixed']
    #             sources = chunk['sources']
                
    #             # Save mixed audio
    #             torchaudio.save(
    #                 samples_dir / f"{song}_mixed.wav",
    #                 mixed.unsqueeze(0), 16000
    #             )
                
    #             # Save individual voices
    #             voice_names = ["soprano", "alto", "tenor", "bass"]
    #             for i, voice in enumerate(voice_names):
    #                 torchaudio.save(
    #                     samples_dir / f"{song}_{voice}.wav",
    #                     sources[i].unsqueeze(0), 16000
    #                 )
                
    #             saved_songs.add(song)
    #             sample_count += 1
    #             print(f"  Saved audio samples for song: {song}")
    
    # print(f"✅ Audio samples saved to: {samples_dir}")
    # print("   You can now listen to the mixed and individual voice files!")
    pass

def preprocess_satb_data():
    """Simple SATB preprocessing for ConvTasNet fine-tuning."""
    
    data_dir = Path("ChoralSingingDataset/ChoralSingingDataset")
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    target_sr = 16000
    chunk_duration = 4.0  # seconds
    chunk_samples = int(chunk_duration * target_sr)
    
    songs = ["ER", "LI", "ND"]  # Process all three songs
    voices = ["soprano", "alto", "tenor", "bass"]
    
    # Process each song with all singer combinations
    processed_data = []
    
    for song in songs:
        print(f"Processing song: {song}")
        
        # Find how many singers we have for each voice (should be 4)
        singers_per_voice = []
        for voice in voices:
            count = 0
            for singer_num in range(1, 5):  # Check singers 1-4
                audio_file = data_dir / f"CSD_{song}_{voice}_{singer_num}.wav"
                if audio_file.exists():
                    count += 1
            singers_per_voice.append(count)
        
        max_singers = min(singers_per_voice)  # Use minimum available
        print(f"  Found {max_singers} singers per voice")
        
        # Create all combinations of singers
        for soprano_singer in range(1, max_singers + 1):
            for alto_singer in range(1, max_singers + 1):
                for tenor_singer in range(1, max_singers + 1):
                    for bass_singer in range(1, max_singers + 1):
                        
                        singer_combo = [soprano_singer, alto_singer, tenor_singer, bass_singer]
                        print(f"  Processing combination: S{soprano_singer}A{alto_singer}T{tenor_singer}B{bass_singer}")
                        
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
                            waveform = torch.mean(waveform, dim=0)  # Convert to mono
                            waveform = waveform / torch.max(torch.abs(waveform))  # Normalize
                            voice_audios.append(waveform)
                        
                        if len(voice_audios) != 4:
                            continue
                            
                        # Find minimum length
                        min_length = min(len(audio) for audio in voice_audios)
                        voice_audios = [audio[:min_length] for audio in voice_audios]
                        
                        # Create mix (simple sum)
                        mixed = sum(voice_audios)
                        mixed = mixed / torch.max(torch.abs(mixed))  # Normalize mix
                        
                        # Split into 4-second chunks
                        for start in range(0, min_length - chunk_samples + 1, chunk_samples):
                            end = start + chunk_samples
                            
                            mixed_chunk = mixed[start:end]
                            source_chunks = torch.stack([audio[start:end] for audio in voice_audios])
                            
                            # Save chunk
                            chunk_data = {
                                'mixed': mixed_chunk,
                                'sources': source_chunks,
                                'song': song,
                                'singer_combo': singer_combo
                            }
                            processed_data.append(chunk_data)
    
    # Split data: ER+LI for train, ND for validation
    train_data = [d for d in processed_data if d['song'] in ['ER', 'LI']]
    val_data = [d for d in processed_data if d['song'] == 'ND']
    
    # Save processed data
    torch.save(train_data, output_dir / "train_data.pt")
    torch.save(val_data, output_dir / "val_data.pt")
    
    print(f"✅ Preprocessing complete!")
    print(f"   Train chunks: {len(train_data)}")
    print(f"   Validation chunks: {len(val_data)}")
    print(f"   Saved to: {output_dir}")

if __name__ == "__main__":
    preprocess_satb_data()
    # save_audio_samples()  # Commented out
