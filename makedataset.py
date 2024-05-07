import os
import numpy as np
import librosa
from torch.utils.data import Dataset
import os
import numpy as np
from torch.utils.data import Dataset
from torchaudio import load

class CustomAudioDataset(Dataset):
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

class ForAudioDataset(Dataset):
    def __init__(self, root):
        self.root = root
        self.classes, self.class_to_idx = self._extract_classes()
        self.samples = self._make_dataset()

    def _extract_classes(self):
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    

    def _is_valid_fp(self, file_name):
        _, ext = os.path.splitext(file_name)
        if ext.lower() not in ('.mp3', '.wav', '.ogg'):
            return True 
    
        return False

    def _make_dataset(self):
        samples = []
        for target_class in self.classes:
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)

            for root_dir, _, file_names in os.walk(target_dir):
                for file_name in file_names:
                    if self._is_valid_fp(file_name):
                        file_path = os.path.join(root_dir, file_name)
                        samples.append((file_path, class_index))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, class_index = self.samples[idx]
        # Load the audio file and preprocess
        waveform, _ = load(audio_path)
        
        wav_len = len(waveform)
        if wav_len >= self.max_len:
            return waveform[:self.max_len]
    
        pad_len = self.max_len - wav_len
        padded_x = np.pad(waveform, (0, pad_len), mode='constant', constant_values=0)
        
        return padded_x, class_index

    def _preprocess_audio(self, waveform):
        waveform = waveform.numpy()[0]  # Convert tensor to numpy array
        max_len = 64600
        if waveform.shape[0] >= max_len:
            return waveform[:max_len]
        else:
            num_repeats = int(max_len / waveform.shape[0]) + 1
            padded_waveform = np.tile(waveform, (1, num_repeats))[:, :max_len][0]
            return padded_waveform