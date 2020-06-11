import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
import argparse

parser = argparse.ArgumentParser(description='Properties of VGG19')

parser.add_argument('-e','--epochs', default = 7, type = int,  help = 'Number of total epochs to run (default: 7)')
parser.add_argument('-b', '--batch_size', default = 64, type = int, help = 'Batch Size (default: 64)')
parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, help='Learning Rate (default: 0.001)')
parser.add_argument('-p', '--print_every', default=20, type=int, help='print frequency (default: 20)')
parser.add_argument('--hidden', default = 1024, type = int, help = 'Number of hidden layers (default: 1024)')
parser.add_argument('--directory_save', type=str, help='Set directory to save checkpoint', default='./')


def train_transforms(train_dir):
    train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225]),
                                          ]),
    return datasets.ImageFolder(train_dir, transform = train_transforms)


def test_transforms(data_transform):
    test_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                          transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225]),
                                         ]),
    
    return datasets.ImageFolder(data_transform, transform = test_transforms)


def trainloader(traindata):
    return torch.utils.data.DataLoader(traindata,batch_size = args.batch_size, shuffle = True)


def testloader(testdata):
    return torch.utils.data.DataLoader(testdata,batch_size = args.batch_size)

def train_model(arch = 'vgg19'):
        model = models.vgg19(pretrained=True)
        print('Using VGG19')
        
        for param in model.parameters():
            param.requires_grad = False
            
        return model

def model_classifier(model, hidden):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(25088, hidden)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden,102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))
    return classifier
    
    
def network(model, criterion, optimizer, print_every, epochs, train_loader, valid_loader, device):
    steps = 0
    model.to(device)
    for i in range(epochs):
        trainloss = 0
        for inputs, labels in train_loader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            trainloss += loss.item()

            if steps % print_every == 0:
                model.eval()
                valid_loss = 0
                acc = 0
                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)
                        valid_loss += criterion(log_ps, labels).item()
                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels.view(*top_class.shape)
                        acc += torch.mean(equal.type(torch.FloatTensor)).item()
                print(f"Epoch {i+1}/{epochs}.. "
                      f"Loss: {trainloss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(dataloaders['validloader']):.3f}.. "
                      f"Accuracy: {acc/len(dataloaders['validloader']):.3f}")
                trainloss = 0
                model.train()
                
    return model


def testmodel(trained_model, device, test_loader):
    #Do validation on the test set
    test_loss = 0
    acc = 0
    model.eval()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        log_ps = model.forward(inputs)
        valid_loss += criterion(log_ps, labels).item()
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equal = top_class == labels.view(*top_class.shape)
        acc += torch.mean(equal.type(torch.FloatTensor)).item()

    print(f"Test Loss: {test_loss/len(test_loader):.3f}.. "
          f"Accuracy: {acc/len(test_loader):.3f}")
    

def saved_model(trained_model, train_data, saved):
    # Save the checkpoint 
    trained_model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'trained_model': 'vgg19',
        'classifier': model.classifier,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
    }
    torch.save(checkpoint, saved + 'Checkpoint.pth')

    

def main():
    global args
    args = parser.parse_args()
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transforms(train_dir)
    valid_data = test_transforms(valid_dir)
    test_data = test_transforms(test_dir)
    
    train_loader = trainloader(train_data)
    valid_loader = trainloader(valid_data)
    test_loader =  testloader(test_data)
    
    model = train_model(arch = 'vgg19')
    
    # Use GPU if it's available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    model.classifier = model_classifier(model, hidden = args.hidden)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    
    trained_model = network(model, train_loader, valid_loader, device, criterion, optimizer, print_every = args.print_every, epochs = args.epochs)
    
    tested_model = testmodel(trained_model, device, test_loader)
    
    save_checkpoint = saved_model(trained_model, train_data, optimizer, saved = args.directory_save)
    
    if __name__ == '__main__': main()
        