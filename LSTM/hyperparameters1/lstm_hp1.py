import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-2


device = 'cpu'
if torch.cuda.is_available():
    print('using gpu')
    device = 'cuda'

model = timm.create_model('sequencer2d_m.in1k', pretrained=False, num_classes=5).to(device)
criterion = nn.CrossEntropyLoss().to(device)

# ======== CHANGE THIS =========================================================
optimizer = optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999))
# ==============================================================================

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=True)

def get_data():
    data_dir = 'Lung Disease Dataset/'

    train_set = datasets.ImageFolder(data_dir + 'train', transform=transform)
    val_set = datasets.ImageFolder(data_dir + 'val', transform=transform)
    test_set = datasets.ImageFolder(data_dir + 'test', transform=transform)

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
    test = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

    return train, val, test

classes = ('Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia')
train, val, test = get_data()


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
print(f"F1-Score: {f1_score(trueValues.cpu(), predictions.cpu(), average='weighted')}")
