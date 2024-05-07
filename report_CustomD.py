import os
import torch
import librosa
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
from model import Model
import numpy as np
from tqdm import tqdm
from makedataset import AudioDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                  
print('Device: {}'.format(device))


model = Model(None, device=device)

checkpoint = torch.load("best.pth", map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# Directories
fake_dir = "Dataset_Speech_Assignment/Dataset_Speech_Assignment/Fake"
real_dir = "Dataset_Speech_Assignment/Dataset_Speech_Assignment/Real"

# Dataset
fake_dataset = AudioDataset(dir=fake_dir)
real_dataset = AudioDataset(dir=real_dir)

# Dataloader
fake_loader = DataLoader(fake_dataset, batch_size=64, shuffle=True)
real_loader = DataLoader(real_dataset, batch_size=64, shuffle=True)

predictions = []
ground_truth = []

# Loop through dataloaders for real and fake data
for loader, label in [(real_loader, 0), (fake_loader, 1)]:
    for processed_audio in loader:
        processed_audio_tensor = processed_audio.float()

        with torch.no_grad():
            output = model(processed_audio_tensor)
            prediction = output[:, 1].item() 

        predictions.append(prediction)
        ground_truth.append(label)

# Calculate AUC
auc_score = roc_auc_score(ground_truth, predictions)

# Calculate ROC curve
fpr, tpr, _ = roc_curve(ground_truth, predictions)

eer = 1.0
for i in range(len(fpr)):
    if fpr[i] >= 1 - tpr[i]:
        eer = fpr[i]
        break

print("AUC:", auc_score)
print("EER:", eer)
