import torch
from sklearn.metrics import roc_auc_score, roc_curve
from model import Model
import numpy as np
from makedataset import AudioDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                  

# Directories
fake_dir = "Dataset_Speech_Assignment/Dataset_Speech_Assignment/Fake"
real_dir = "Dataset_Speech_Assignment/Dataset_Speech_Assignment/Real"

# Dataset
fake_dataset = AudioDataset(dir=fake_dir)
real_dataset = AudioDataset(dir=real_dir)

# Dataloader
fake_loader = DataLoader(fake_dataset, batch_size=64, shuffle=True)
real_loader = DataLoader(real_dataset, batch_size=64, shuffle=True)

class AudioClassifierEvaluator:
    def __init__(self, model_filename, real_loader, fake_loader, device):
        self.model = self.load_model(model_filename, device)
        self.real_loader = real_loader
        self.fake_loader = fake_loader
        self.device = device

    def load_model(self, model_filename, device):
        model = Model(None, device=device)  # Assuming Model is defined somewhere
        checkpoint = torch.load(model_filename, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model

    def evaluate(self):
        predictions = []
        ground_truth = []

        for loader, label in [(self.real_loader, 0), (self.fake_loader, 1)]:
            for processed_audio in loader:
                processed_audio_tensor = processed_audio.float().to(self.device)

                with torch.no_grad():
                    output = self.model(processed_audio_tensor)
                    prediction = output[:, 1].item()

                predictions.append(prediction)
                ground_truth.append(label)

        auc_score = roc_auc_score(ground_truth, predictions)
        fpr, tpr, _ = roc_curve(ground_truth, predictions)
        
        eer = 1.0
        for i in range(len(fpr)):
            if fpr[i] >= 1 - tpr[i]:
                eer = fpr[i]
                break

        print("AUC:", auc_score)
        print("EER:", eer)

        return auc_score, eer
    

evaluator = AudioClassifierEvaluator("best.pth", real_loader, fake_loader, device)
auc_score, eer = evaluator.evaluate()