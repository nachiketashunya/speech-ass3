from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import numpy as np
import torch
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, lr=5e-5, num_epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.lr = lr
        self.num_epochs = num_epochs
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([0.1, 0.9]).to(device))

    def train_epoch(self):
        running_loss = 0.0
        num_total = 0.0
        num_batches = len(self.train_loader)
        self.model.train()

        progress_bar = tqdm(self.train_loader, desc='Training', leave=False)
        
        for batch_x, batch_y in progress_bar:
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.view(-1).type(torch.int64).to(self.device)
            
            self.optimizer.zero_grad()
            
            batch_out = self.model(batch_x)
            batch_loss = self.criterion(batch_out, batch_y)
            
            running_loss += batch_loss.item() * batch_size
            
            batch_loss.backward()
            self.optimizer.step()

            progress_bar.set_postfix(loss=running_loss / num_batches)

        progress_bar.close()
        running_loss /= len(self.train_loader.dataset)

        wandb.log({"train_loss": running_loss})

        return running_loss

    def validate(self):
        running_loss = 0.0
        num_total = 0.0
        num_batches = len(self.val_loader)
        self.model.eval()
        progress_bar = tqdm(self.val_loader, desc='Validation', leave=False)
        with torch.no_grad():
            for batch_x, batch_y in progress_bar:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.view(-1).type(torch.int64).to(self.device)
                batch_out = self.model(batch_x)
                batch_loss = self.criterion(batch_out, batch_y)
                running_loss += batch_loss.item() * batch_size
                progress_bar.set_postfix(loss=running_loss / num_batches)

        progress_bar.close()
        running_loss /= len(self.val_loader.dataset)
        wandb.log({"val_loss": running_loss})
        return running_loss
    
    def eval(self, test_loader):
        true_labels = []
        predicted_scores = []

        self.model.eval()

        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Testing"):
                output = self.model(data)
                predicted_scores.extend(output[:, 1].cpu().numpy())
                true_labels.extend(target.cpu().numpy())

        true_labels = np.array(true_labels)
        predicted_scores = np.array(predicted_scores)

        auc_score = roc_auc_score(true_labels, predicted_scores)

        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        threshold = interp1d(fpr, thresholds)(eer)

        print("AUC:", auc_score)
        print("EER:", eer)
        print("Threshold at EER:", threshold)

        return auc_score, eer, threshold

    def run(self, project, run_name):
        wandb.init(project=project, name=run_name)

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            print(f"Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        wandb.finish()

