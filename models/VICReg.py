import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import sys
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torchinfo import summary


FILE  = "../saved_models/VICReg.pth"

#Hyper Parameters
batch_size = 128
learning_rate = 0.0001
epochs = 100
# Dimension (D) of the representations
embedding_dimension = 32
lam = 25
mu = 25
nu = 1


dataset = torchvision.datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
train_data, val_data = random_split(dataset, [50000, 10000])

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
n_total_steps = len(train_loader)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VICReg(nn.Module):
    def __init__(self):
        super(VICReg, self).__init__()
        #32 output channels --> change kernel size to 3 or 5
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1)
        #64 output channels --> 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16,32, kernel_size=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 64)
        # 3 fully connected layers
        self.fc3 = nn.Linear(64,embedding_dimension)
        self.fc1_classify = nn.Linear(128,64)
        self.fc2_classfiy = nn.Linear(64,64)
        self.fc3_classify = nn.Linear(64,10)

    def forward(self, x):
        if (arg1 == "Train"):
            # print(f"Starting training of VICReg Model with {embedding_dimension} embedding dimension")
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
        elif (arg1 == "Classify"):
            x = F.relu(self.conv1(x))
            x = self.max_pool(x)
            x = F.relu(self.conv2(x))
            x = self.max_pool(x)
            x = F.relu(self.conv3(x))
            x = self.max_pool(x)
            # print(f"Shape of representation space {x.shape}")
            ## Representation Space
            x = x.view(-1,128)
            # print(f"Shape after flatenning {x.shape}")
            x = F.relu(self.fc1_classify(x))
            x = F.relu(self.fc2_classfiy(x))
            x = self.fc3_classify(x)
            # print(f"Shape after entire Network {x.shape}")
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
    def VIC_Reg_loss(self, aug1, aug2, model):
        # print(type(aug1))
        # print(type(aug2))
        output_i = model(aug1)
        output_j = model(aug2)
        # print(type(output_i))
        # print(type(output_j))

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
        cov_loss = (model.off_diagonal(cov_output_i).pow(2).sum() / embedding_dimension) + (model.off_diagonal(cov_output_j).pow(2).sum() / embedding_dimension)

        # compute loss between two different data augs!
        # train for 20 epochs and make a loss curve
        std_loss = mu * std_loss
        cov_loss = nu * cov_loss
        loss = sim_loss + std_loss + cov_loss

        return loss, sim_loss, std_loss, cov_loss
    def produce_two_augs(self, image, model):
        image_i = model.data_aug(image)
        image_j = model.data_aug(image)
        
        return image_i, image_j
  
    def before_train(self, model):
        with torch.no_grad():
            image_train = next(iter(train_loader))[0]
            image_train = image_train.to(device)
            image_val = next(iter(val_loader))[0]
            image_val = image_val.to(device)
            aug1, aug2 = model.produce_two_augs(image_train, model)
            loss_b, sim_loss_b, std_loss_b, cov_loss_b = model.VIC_Reg_loss(aug1, aug2,model)
            val_loss_b, _, _, _, = model.VIC_Reg_loss(aug1, aug2, model)
            print(loss_b.item())
            print(val_loss_b.item())
            f = open("../csv's/VICReg_metrics_64_space.csv", "a")
            f.write(f"{0},{loss_b.item()}, {val_loss_b.item()}, {sim_loss_b}, {std_loss_b}, {cov_loss_b}\n")
            f.close()

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)
        self.conv3 = nn.Conv2d(16,32, kernel_size=2)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        x = x.view(-1,32)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

arg1 = sys.argv[1]
if (arg1 == "Train"):
    model_vicreg = VICReg().to(device=device)
    print(summary(model_vicreg, input_size=[(batch_size, 1, 20, 20)]))
    optimizer = torch.optim.Adam(model_vicreg.parameters(), lr = learning_rate)

    f = open("../csv's/VICReg_metrics_64_space.csv","w+" )
    f.write("Epoch, Train Loss, Val Loss, Inv Loss, Var Loss, Cov Loss\n")
    f.close()

    model_vicreg.before_train(model_vicreg)

    for epoch in range(epochs):
        running_loss = 0.0
        running_inv_loss = 0.0
        running_var_loss = 0.0
        running_cov_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # two randomly augmented versions of image
            image_i, image_j = model_vicreg.produce_two_augs(images, model_vicreg)
            # Calculate VICReg Losss
            loss, sim_loss, std_loss, cov_loss = model_vicreg.VIC_Reg_loss(image_i, image_j, model_vicreg)
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
        running_val_loss = 0.0
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            image_i, image_j = model_vicreg.produce_two_augs(images, model_vicreg)
            loss_val, sim_loss_val, std_loss_val, cov_loss_val = model_vicreg.VIC_Reg_loss(image_i, image_j, model_vicreg)

            running_val_loss += loss_val.item()
        # depending on # of epochs, save the model
        torch.save(model_vicreg.state_dict(), FILE)


        epoch_loss = running_loss / len(train_loader)
        epoch_inv_loss = running_inv_loss / len(train_loader)
        epoch_var_loss = running_var_loss / len(train_loader)
        epoch_cov_loss = running_cov_loss / len(train_loader)
        epoch_val_loss = running_val_loss / len(val_loader)
        f = open("../csv's/VICReg_metrics_64_space.csv", "a")
        f.write(f"{epoch + 1}, {epoch_val_loss}, {epoch_loss}, {epoch_inv_loss}, {epoch_var_loss}, {epoch_cov_loss}\n")
        f.close()
elif (arg1 == "Classify"):
    loaded_model = VICReg().to(device=device)
    loaded_model.load_state_dict(torch.load(FILE))
    loaded_model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(loaded_model.parameters(), lr=learning_rate)

    f = open("../csv's/Classifer_metrics.csv","w+" )
    f.write("Epoch, Train Loss, Val Loss, Train Accuracy, Val Accuracy\n")
    f.close()
    print("Loaded the model")

    with torch.no_grad():
        image_train = next(iter(train_loader))[0]
        image_train = image_train.to(device)
        train_label = next(iter(train_loader))[1]
        train_label = train_label.to(device)
        image_val = next(iter(val_loader))[0]
        image_val = image_val.to(device)
        val_label = next(iter(val_loader))[1]
        val_label = val_label.to(device)
        output_train = loaded_model(image_train)
        output_val = loaded_model(image_val)
        loss_t = criterion(output_train,train_label)
        loss_v = criterion(output_val, val_label)

    print(f"Loss before training: {loss_t.item()}")
    print(f"Loss before validation: {loss_v.item()}")
    f = open("../csv's/Classifer_metrics.csv","a")
    f.write(f"{0},{loss_t.item()}, {loss_v.item()}\n")
    f.close()

    for epoch in range(epochs):
        running_trainloss = 0.0
        running_val_loss = 0.0
        n_correct = 0.0
        n_samples = 0.0
        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = loaded_model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            train_loss = criterion(outputs, labels)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            running_trainloss += train_loss.item()

            if (i+1) % 100 ==0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {train_loss.item():.4f}')
        train_acc = 100.0 * n_correct / n_samples

        with torch.no_grad():
            n_correct = 0.0
            n_samples = 0.0
            for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = loaded_model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

                val_loss = criterion(outputs, labels)

                running_val_loss += val_loss.item()
            
            epoch_train_loss = running_trainloss / len(train_loader)
            epoch_val_loss = running_val_loss / len(val_loader)
            val_acc = 100.0 * n_correct / n_samples
            f = open("../csv's/Classifer_metrics.csv","a")
            f.write(f"{epoch+1},{epoch_train_loss},{epoch_val_loss},{train_acc},{val_acc}\n")
            f.close()

else:
    print("Error: Invalid arguments passed")
