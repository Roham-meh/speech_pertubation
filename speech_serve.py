import torchaudio
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchaudio.transforms as transforms
import torchattacks
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


original_dataset = torchaudio.datasets.SPEECHCOMMANDS(root="./", download=True)


labels = [label for _, _, label, _, _ in original_dataset]

label_to_index = {label: index for index, label in enumerate(sorted(set(labels)))}
num_classes = len(label_to_index)

class MelSpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_length=128):
        self.dataset = dataset
        self.mel_spectrogram = transforms.MelSpectrogram(normalized=True)
        self.target_length = target_length 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, _, label, _, _ = self.dataset[idx]
        mel_spec = self.mel_spectrogram(waveform)

        # Check and modify the size of mel_spec to match target_length
        current_length = mel_spec.shape[2]
        if current_length < self.target_length:
            padding_amount = self.target_length - current_length
            mel_spec = torch.nn.functional.pad(mel_spec, (0, padding_amount))
        elif current_length > self.target_length:
            mel_spec = mel_spec[:, :, :self.target_length]

        label_index = label_to_index[label] 
        return mel_spec, label_index
    
mel_dataset = MelSpectrogramDataset(original_dataset)



train_size = int(0.8 * len(mel_dataset))  
test_size = len(mel_dataset) - train_size  
train_dataset, test_dataset = random_split(mel_dataset, [train_size, test_size])  


model = models.resnet18(pretrained=True)  # Pretrained ResNet18 model
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)


criterion = nn.CrossEntropyLoss()  
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)  

device = next(model.parameters()).device
print(device)




for epoch in range(1):
    model.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/1, Batch {i+1}/{len(train_loader)}")

torch.save(model.state_dict(), 'Documents/model_state_dict.pth')
#####





model.eval()  
correct = 0  
total = 0  
with torch.no_grad():  
    for data in test_loader:  
        inputs, labels = data  
        outputs = model(inputs)  
        _, predicted = torch.max(outputs.data, 1)  
        total += labels.size(0)  
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on test images: {100 * correct // total}%')


epsilon = 0.3 
model.eval()

correct = 0
total = 0

for data, target in test_loader:
    data.requires_grad = True

    # Forward pass
    output = model(data)
    loss = F.nll_loss(output, target)

    # Backward pass
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data

    # FGSM
    perturbed_data = data + epsilon * data_grad.sign()
    perturbed_output = model(perturbed_data)

    # Get predicted class from the perturbed output
    _, predicted = torch.max(perturbed_output.data, 1)
    total += target.size(0)
    correct += (predicted == target).sum().item()


accuracy = 100 * correct / total
print(f'Accuracy after FGSM attack: {accuracy}%')



model.eval()


epsilon = 0.3  # Maximum perturbation
alpha = 0.01  
iterations = 40 

# Creating PGD attack
pgd_attack = torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=iterations)l
correct = 0
total = 0
for data, target in test_loader:
    # Apply attack
    adversarial_data = pgd_attack(data, target)

    # Evaluate on adversarial examples
    output = model(adversarial_data)
    _, predicted = torch.max(output.data, 1)

    total += target.size(0)
    correct += (predicted == target).sum().item()

# Accuracy
accuracy = 100 * correct / total
print(f'Accuracy after PGD attack: {accuracy}%')
