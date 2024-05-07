import os
import numpy as np
import librosa
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, dir, max_len=64600):
        self.max_len = max_len
        self.paths = os.listdir[dir]

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.preprocess_audio(self.paths[idx])
    
    def pad_audio(self, x):
        x_len = len(x)
        if x_len >= self.max_len:
            return x[:self.max_len]
    
        pad_len = self.max_len - x_len
        padded_x = np.pad(x, (0, pad_len), mode='constant', constant_values=0)
        return padded_x
    
    def preprocess_audio(self, audio_path):
        _, ext = os.path.splitext(audio_path)
        if ext.lower() not in ('.mp3', '.wav'):
            return None
        
        audio, sr = librosa.load(audio_path, sr=None)
        audio = self.pad(audio)

        return audio
