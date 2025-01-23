#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:41:46 2025

@author: alfred
"""

'''
This python code analyzes BCI Competition IV 2A with EEGnet by collecting epochs
with mne module. It is necessary with installing numpy, mne, torch.
'''

import mne, torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

#%% Collect Channels & EOG names
ch_name = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', \
           'C2', 'C4', 'C6','CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    

eog_names = ['EOG-left', 'EOG-central', 'EOG-right']

#%% Load EEG file with mne module. But, the mne module cannot read the names of channel
file_path = './A01T.gdf'
raw = mne.io.read_raw_gdf(file_path, preload=True, eog=eog_names)
raw.info

#%% Therefore, insert channel's name individually
mapping = {'EEG-Fz': 'Fz',
           'EEG-0': 'FC3',
           'EEG-1': 'FC1',
           'EEG-2': 'FCz',
           'EEG-3': 'FC2',
           'EEG-4': 'FC4',
           'EEG-5': 'C5',
           'EEG-C3': 'C3',
           'EEG-6': 'C1',
           'EEG-Cz': 'Cz',
           'EEG-7': 'C2',
           'EEG-C4': 'C4',
           'EEG-8': 'C6',
           'EEG-9': 'CP3',
           'EEG-10': 'CP1',
           'EEG-11': 'CPz',
           'EEG-12': 'CP2',
           'EEG-13': 'CP4',
           'EEG-14': 'P1',
           'EEG-Pz': 'Pz',
           'EEG-15': 'P2',
           'EEG-16': 'POz',
          }

mne.rename_channels(raw.info, mapping)

eeg = raw.copy().pick(ch_name) # Copy the signal and insert name

events, event_id = mne.events_from_annotations(eeg) # Load EEG events and id from signal

#%% Set hyper parameter. You can select time
tmin, tmax = 0.5, 2.5

epochs = mne.Epochs(eeg, events, event_id={'769':7, '770':8, '771':9, '772':10},
                    tmin=tmin, tmax=tmax, baseline=None, preload=True)

X = epochs.get_data(copy=True) # This means input signal
y = epochs.events[:, -1]-7 # This means output signal

#%% Define dataset class
class EEGDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

#%% Split dataset into training
from sklearn.model_selection import StratifiedShuffleSplit
split_ratio=0.8
splitter = StratifiedShuffleSplit(n_splits=1, test_size=1-split_ratio, random_state=42)

for train_idx, val_idx in splitter.split(X, y):
    train_dataset = X[train_idx]
    valid_dataset = X[val_idx]
    train_labels = y[train_idx]
    valid_labels = y[val_idx]

train_data = EEGDataset(train_dataset, train_labels)
valid_data = EEGDataset(valid_dataset, valid_labels)

#%% Load them into DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

#%% Define EEGNet
class EEGNet(nn.Module):
    def __init__(self, nb_classes=4, Chans=22, Samples=501, dropoutRate=0.25,
                 kernLength=63, F1=8, D=2, F2=16, dropoutType='Dropout'):
        
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_classes = nb_classes
        self.kL = kernLength
        
        self.block1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, self.kL), padding='same', bias=False),
            nn.BatchNorm2d(self.F1),
            nn.Conv2d(self.F1, self.F1 * self.D, (Chans, 1), groups=self.F1, padding=0),
            nn.BatchNorm2d(self.F1 * self.D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropoutRate)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, kernLength//4), padding='same', groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropoutRate)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2*(Samples // 32), nb_classes),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x

#%% Start training & Testing
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model = EEGNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    val_correct = 0
    val_total = 0
    
    for data, labels in train_loader:
        data, labels = data.unsqueeze(1).to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    with torch.no_grad():
        for data, labels in valid_loader:
            data, labels = data.unsqueeze(1).to(device), labels.to(device)
            outputs = model(data)
            
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    val_acc = val_correct / val_total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Validation: {val_correct} / {val_total}')

print("Training complete.")

#%% File saving
torch.save(model, './EEG_model.pt')

