import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import numpy as np

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
DROPOUT = 0.25

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# process and split data from data set
def get_data():
    data_dir = 'Lung Disease Dataset/'

    train_set = datasets.ImageFolder(data_dir + 'train', transform=transform)
    val_set = datasets.ImageFolder(data_dir + 'val', transform=transform)
    test_set = datasets.ImageFolder(data_dir + 'test', transform=transform)

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train, val, test


# classification labels
classes = ('Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia')
train, val, test = get_data()


# ZF-Net (CNN Architecture)
# Implementation reference from https://github.com/CellEight/Pytorch-ZFNet
# and "Visualizing and Understanding Convolutional Networks" https://arxiv.org/pdf/1311.2901
class ZFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convLayers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            nn.LocalResponseNorm(size=5,alpha=10e-4,beta=0.75,k=2.0),

            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
        )

        self.fcLayers = nn.Sequential(
            nn.Linear(9216,4096),
            nn.Dropout(DROPOUT),
            nn.ReLU(),

            nn.Linear(4096,4096),
            nn.Dropout(DROPOUT),
            nn.ReLU(),

            nn.Linear(4096,5),
        )

    def forward(self, x):
        x = self.convLayers(x)
        x = x.view(x.size(0), -1)
        x = self.fcLayers(x)
        return x


# device selection and model, loss function, optimizer initialization
device = 'cpu'
if torch.cuda.is_available():
    print('using gpu')
    device = 'cuda'

model = ZFNet().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))

# Train the Model
for epoch in range(EPOCHS):
    model.train()
    
    trainingLoss = 0.0
    validationLoss = 0.0
    validationPredictions = torch.tensor([]).to(device)
    validationTrueValues = torch.tensor([]).to(device)
    
    for i, data in enumerate(train, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        trainingLoss += loss.item()

    # Validate using validation set
    with torch.no_grad():
        model.eval()

        for i, data in enumerate(val, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validationLoss += loss.item()
            validationPredictions = torch.cat((validationPredictions, torch.argmax(outputs, dim=1)), dim=0)
            validationTrueValues = torch.cat((validationTrueValues, labels), dim=0)

    trainSteps = len(train.dataset) / BATCH_SIZE
    valSteps = len(val.dataset) / BATCH_SIZE
    f1 = f1_score(validationTrueValues.cpu(), validationPredictions.cpu(), average='weighted')

    print(f"Epoch {epoch + 1} Average Training Loss: {round(trainingLoss / trainSteps, 6)} Average Validation Loss: {round(validationLoss / valSteps, 6)} Validation F1-Score: {round(f1, 6)}")

            
# Get loss from test set after training
testLoss = 0.0
predictions = torch.tensor([]).to(device)
trueValues = torch.tensor([]).to(device)

with torch.no_grad():
    model.eval()

    for i, data in enumerate(test, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        testLoss += loss.item()
        predictions = torch.cat((predictions, torch.argmax(outputs, dim=1)), dim=0)
        trueValues = torch.cat((trueValues, labels), dim=0)

testSteps = len(test.dataset) / BATCH_SIZE
print(f"Average Test Loss: {round(testLoss / testSteps, 6)}")
print(f"Test set F1-Score: {f1_score(trueValues.cpu(), predictions.cpu(), average='weighted')}")
