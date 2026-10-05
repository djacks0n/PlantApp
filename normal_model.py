from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from dataset import get_dataloaders, NUM_CLASSES

train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
criterion = nn.CrossEntropyLoss()

model = resnet50(weights=ResNet50_Weights.DEFAULT)

for param in model.parameters():
    param.requires_grad = False

num_plant_species = 16 #num of plant unique plant species in the dataset
model.fc = nn.Linear(model.fc.in_features, num_plant_species)
