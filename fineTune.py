import torch
from makedataset import ForAudioDataset
import os
from torch.utils.data import DataLoader
from model import Model
import torch
from trainer import Trainer

# Define the root directory where your data is stored
root = "/teamspace/studios/this_studio/SpeechAssign3/for-2seconds"  # Replace this with the path to your data folder

# Create datasets
train_dataset = ForAudioDataset(root=os.path.join(root, "training"))
test_dataset = ForAudioDataset(root=os.path.join(root, "testing"))
validation_dataset = ForAudioDataset(root=os.path.join(root, "validation"))

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=6)

#GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
model = Model(None, device=device)

model.load_state_dict(torch.load('/teamspace/studios/this_studio/final_model.pth'))

trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, device=device)

trainer.run(project="SUASS3", run_name="FoR Fine Tune")

