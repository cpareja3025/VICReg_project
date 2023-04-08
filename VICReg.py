import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchinfo import summary

#Hyper Parameters
batch_size = 64
learning_rate = 0.0001
epochs = 100
# Dimension (D) of the representations
embedding_dimension = 32
lam = 1
mu = 0.1
nu = 1e-09

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=0.1307, std=0.3081)])
train_dataset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_augs(nn.Module):
    def __init__(self):
        super(CNN_augs, self).__init__()
        #32 output channels --> change kernel size to 3 or 5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=1)
        #64 output channels --> 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(5184, 256)
        self.fc2 = nn.Linear(256, 128)
        # 3 fully connected layers
        # embedding dimension will be 32
        self.fc3 = nn.Linear(128,embedding_dimension)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = x.view(-1,5184)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def data_aug(self, img_tensor):
        aug = transforms.RandomResizedCrop(20, scale=(0.08,0.1))(img_tensor)
        aug = transforms.RandomHorizontalFlip(p=0.5)(aug)
        aug = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)(aug)
        #aug = transforms.RandomGrayscale(0.2)(aug)
        aug = transforms.GaussianBlur(kernel_size=23, sigma=0.5)(aug)
        aug = transforms.RandomSolarize(threshold=0.3,p=0.1)(aug)
        #aug = transforms.Normalize()(aug)
        return aug
    def off_diagonal(self, x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
        
    

model_vicreg = CNN_augs().to(device=device)
optimizer = torch.optim.Adam(model_vicreg.parameters(), lr = learning_rate)

f = open("/Users/cpare/repos/VICReg_project/csv's/VICReg_metrics2.csv","w+" )
f.write("Epoch, Loss\n")
f.close()
n_total_steps = len(train_loader)
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        # two randomly augmented versions of image
        image_i = model_vicreg.data_aug(images)
        image_j = model_vicreg.data_aug(images)

        labels = labels.to(device)
        #compute representations
        output_i = model_vicreg(image_i)
        output_j = model_vicreg(image_j)

        #invariance loss
        sim_loss = nn.MSELoss()
        sim_loss = lam * sim_loss(output_i, output_j)

        #variance loss
        std_output_i = torch.sqrt(torch.var(output_i, dim=0) + 1e-04)
        std_output_j = torch.sqrt(torch.var(output_j, dim = 0) + 1e-04)
        std_loss = torch.mean(F.relu(1-std_output_i)) + torch.mean(F.relu(1-std_output_j))

        #covariance loss
        output_i = output_i - torch.mean(output_i, dim=0)
        output_j = output_j - torch.mean(output_j, dim=0)
        cov_output_i = (torch.matmul(torch.transpose(output_i, 0, 1), output_i) / (batch_size -1))
        cov_output_j = (torch.matmul(torch.transpose(output_j, 0, 1), output_j) / (batch_size -1))
        cov_loss = (model_vicreg.off_diagonal(cov_output_i).pow(2).sum() / embedding_dimension) + (model_vicreg.off_diagonal(cov_output_j).pow(2).sum() / embedding_dimension)

        # compute loss between two different data augs!
        # train for 20 epochs and make a loss curve
        loss = (sim_loss) + (mu * std_loss) + (nu*cov_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * image_i.size(1)

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f} ')
    epoch_loss = running_loss / len(train_loader)
    f = open("/Users/cpare/repos/VICReg_project/csv's/VICReg_metrics2.csv", "a")
    f.write(f"{epoch + 1}, {epoch_loss}\n")
    f.close()