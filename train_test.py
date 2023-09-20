
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from matplotlib import pyplot as plt
import numpy as np
import torchvision
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

train_losses = []
test_losses = []
train_acc = []
test_acc = []

#Unet
def train_unet(model, device, train_loader, optimizer, epoch, train_losses,criterion,scheduler,loss_cr='BCE'):

    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0
    correct = 0

    for batch_idx, (data, target, label_one) in enumerate(pbar):
        # get samples
        label_one = label_one.type(torch.FloatTensor)
        data, target, label_one = data.to(device), target.to(device), label_one.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        if loss_cr == 'BCE':
            loss = criterion(y_pred, label_one)
        else:
            loss = criterion(y_pred, target)

        train_loss += loss.item()

        if loss_cr == 'BCE':
            pred = torch.argmax(y_pred, 1)
        else:
            _, pred = torch.max(y_pred, 1)

        correct += torch.mean((pred == target).type(torch.float64))

        # Backpropagation
        loss.backward()
        optimizer.step() 
    
    train_losses.append(train_loss)

    print(f'Training Loss={train_loss} Accuracy={correct}')

def test_unet(model, device, test_loader,test_losses,criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target, label_one in test_loader:
            label_one = label_one.type(torch.FloatTensor)
            data, target, label_one = data.to(device), target.to(device), label_one.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = torch.argmax(output, 1)
            correct += torch.mean((pred == target).type(torch.float64))

    #test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print(f'Test set: Average loss={test_loss} Accuracy={correct}')
    

def fit_model_unet(model, optimizer, criterion, trainloader, testloader, EPOCHS, device,scheduler=None,loss_cr='BCE'):
    train_losses = []
    test_losses = []
    
    for epoch in range(EPOCHS):
        print("\n EPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train_unet(model, device, trainloader, optimizer, epoch, train_losses, criterion,scheduler,loss_cr=loss_cr)

    return model, train_losses, test_losses

# for binary class
def binary_dice_loss(pred, target):
    smooth = 1e-5
    #pred = F.sigmoid(pred)

    # flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    dice = (2. * intersection + smooth) / (union + smooth)
    
    return 1 - dice   

#for multi-class
def dice_loss(predicted, target, num_classes=3, epsilon=1e-5):
    dice_losses = 0
    
    for class_index in range(num_classes):  # Loop through all classes except background
        
        smooth = 1e-5
        #pred = F.sigmoid(pred)
    
        # flatten predictions and targets
        pred = predicted.view(-1)
        target = (target == class_index).view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        
        dice_loss = 1 - dice   
        dice_losses += dice_loss
    
    # Calculate the average Dice loss for all classes (excluding background)
    return dice_losses / num_classes



def train_vae(model, device, train_loader, optimizer, epoch, train_losses,criterion,scheduler):

    model.train()
    pbar = tqdm(train_loader)
    train_loss = 0

    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        X_hat, mean, logvar = model(data,target)

        # Calculate loss
        reconstruction_loss = criterion(X_hat, data)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean**2)
        loss_tot = reconstruction_loss + KL_divergence

        train_loss += loss_tot.item()

        # Backpropagation
        loss_tot.backward()
        optimizer.step() 
    
    train_losses.append(train_loss/len(train_loader.dataset))

    print(f'\nAverage Training Loss={train_loss/len(train_loader.dataset)}')

def fit_model_vae(model, optimizer, criterion, trainloader, EPOCHS, device,scheduler=None):
    train_losses = []
    
    for epoch in range(EPOCHS):
        print("\nEPOCH: {} (LR: {})".format(epoch+1, optimizer.param_groups[0]['lr']))
        train_vae(model, device, trainloader, optimizer, epoch, train_losses, criterion,scheduler)

    return model, train_losses

def train(model, device, train_loader, optimizer, epoch, sched, criterion, train_acc , train_losses, max_lr):
  
  if sched == 'StepLR':
    scheduler = StepLR(optimizer, step_size=100, gamma=0.25)
    sched_flag = 'X'
  elif sched == 'OneCycle':
    scheduler = OneCycleLR(optimizer=optimizer, max_lr=max_lr, epochs=epoch, steps_per_epoch=len(train_loader), pct_start=5/epoch, div_factor=10,three_phase=False) 
    sched_flag = 'X'
  else:
    sched_fl = ''
    
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    if sched_flag == 'X':
      scheduler.step()

    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)

def test(model, device, test_loader, criterion,test_acc , test_losses):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))
