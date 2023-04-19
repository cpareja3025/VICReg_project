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
batch_size = 128
learning_rate = 0.0001
epochs = 100
# Dimension (D) of the representations
embedding_dimension = 512
lam = 0.1
mu = 0.1
nu = 1e-07

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
        self.conv3 = nn.Conv2d(64,128, kernel_size=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 512)
        self.fc2 = nn.Linear(512, 512)
        # 3 fully connected layers
        self.fc3 = nn.Linear(512,embedding_dimension)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        #print(f"Shape of Representations space after encoder {x.shape}")
        # Shape of Representations space (Y and Y prime) is [64, 128, 1, 1]
        x = x.view(-1,128)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #print(f'Shape of Embedding space after expander {x.shape}')
        # Shape of Embeddings space (Z and Z prime) is [batch_size, embedding_dimension]
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
    def VIC_Reg_loss(self, aug1, aug2):
        output_i = model_vicreg(aug1)
        output_j = model_vicreg(aug2)

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
        std_loss = mu * std_loss
        cov_loss = nu * cov_loss
        loss = sim_loss + std_loss + cov_loss

        return loss, sim_loss, std_loss, cov_loss
    def produce_two_augs(self, image):
        image_i = model_vicreg.data_aug(image)
        image_j = model_vicreg.data_aug(image)
        
        return image_i, image_j
  
    def before_train(self, image):
        aug1, aug2 = model_vicreg.produce_two_augs(image)
        loss, sim_loss, std_loss, cov_loss = model_vicreg.VIC_Reg_loss(aug1, aug2)

        return loss


        
    

model_vicreg = CNN_augs().to(device=device)
print(summary(model_vicreg, input_size=[(batch_size, 1, 20, 20)]))
optimizer = torch.optim.Adam(model_vicreg.parameters(), lr = learning_rate)

f = open("./VICReg_metrics_allLosses_withVal.csv","w+" )
f.write("Epoch, Loss, Inv Loss, Var Loss, Cov Loss\n")
f.close()
n_total_steps = len(train_loader)

with torch.no_grad():
  image = next(iter(train_loader))[0]
  aug1, aug2 = model_vicreg.produce_two_augs(image)
  loss_b, _, _, _, = model_vicreg.VIC_Reg_loss(aug1, aug2)
  print(loss_b.item())


for epoch in range(epochs):
    running_loss = 0.0
    running_inv_loss = 0.0
    running_var_loss = 0.0
    running_cov_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        # two randomly augmented versions of image
        image_i, image_j = model_vicreg.produce_two_augs(images)
        # Calculate VICReg Losss
        loss, sim_loss, std_loss, cov_loss = model_vicreg.VIC_Reg_loss(image_i, image_j)
        # Train
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() 
        running_inv_loss += sim_loss
        running_var_loss += std_loss
        running_cov_loss += cov_loss

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}, Invariance Loss: {sim_loss}, Variance Loss: {std_loss}. Covariance Loss: {cov_loss}')
    epoch_loss = running_loss / len(train_loader)
    epoch_inv_loss = running_inv_loss / len(train_loader)
    epoch_var_loss = running_var_loss / len(train_loader)
    epoch_cov_loss = running_cov_loss / len(train_loader)
    f = open("VICReg_metrics_allLosses_withVal.csv", "a")
    f.write(f"{epoch + 1}, {epoch_loss}, {epoch_inv_loss}, {epoch_var_loss}, {epoch_cov_loss}\n")
    f.close()