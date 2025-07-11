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
        
        for voice in voices:
            processed_audio[song][voice] = {}
            
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
                
                processed_audio[song][voice][singer_num] = waveform
                print(f"  Loaded {song}_{voice}_{singer_num}: {waveform.shape[0]} samples")
    
    return processed_audio

if __name__ == "__main__":
    audio_data = load_and_process_voices()
    print("✅ All audio files loaded and processed")
