import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
import argparse, json


def arg_parser():
    '''
    Initializing argparse for values such as epochs, batch_size, learning_rate as well as print count, hidden_layers, directory to save checkpoint and gpu
    '''
    parser = argparse.ArgumentParser(description='Properties of VGG19')
    #Creating Arguments
    parser.add_argument('--arch', type=str, default = 'vgg19',help='Models architeture. Default is VGG19. Choose from : VGG19: Predefined in_features = 25088 hidden = 1024, Densenet : in_features and hidden must be defined')
    parser.add_argument('--in_features', type = int,default = 25088, action='store', help = 'Number of total in_features Default set for VGG19 - 25088')
    parser.add_argument('-e','--epochs', default = 7, type = int, action='store', help = 'Number of total epochs to run (default: 7)')
    parser.add_argument('--lr', '--learning_rate', default=0.001, type=float, action='store', help='Learning Rate (default: 0.001)')
    parser.add_argument('-p', '--print_every', default=20, type=int, action='store', help='print frequency (default: 20)')
    parser.add_argument('--hidden', default = 1024, type = int, action='store', help = 'Number of hidden layers (default: 1024)')
    parser.add_argument('--directory_save', type=str, help='Set directory to save checkpoint', default='./')
    parser.add_argument('--gpu', action="store", default="gpu", help='USE GPU')
    
    args = parser.parse_args()
    return args


#Dataset Location
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# Locating and Transforming dataset and Normalizing it
def train_transforms(train_dir):

    train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
    return train_data


def test_transforms(test_dir):

    test_data = datasets.ImageFolder(test_dir, transform = test_transforms)
    return test_data


def valid_transforms(valid_dir):

    valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    return valid_data


image_train =  transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])


image_valid = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])


image_test = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                      [0.229, 0.224, 0.225])])



# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir, transform=image_train)
valid_data = datasets.ImageFolder(valid_dir, transform=image_valid)
test_data = datasets.ImageFolder(test_dir, transform=image_test)


# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True)
validloader = torch.utils.data.DataLoader(test_data, batch_size=60, shuffle = True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=60)


print(' \n Dataset Loaded \n')


#Initializing Pre trained model
def train_model(arch):
    model = eval("models.{}(pretrained=True)".format(arch))
    model.name = arch
    for param in model.parameters():
        param.requires_grad = False

    return model


#Building the network
def model_classifier(model, hidden, in_features):
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden,102)),
        ('output', nn.LogSoftmax(dim = 1))
    ]))

    model.classifier = classifier
    print(model)
    return classifier


#Training the network
def network(model, criterion, optimizer, print_every, epochs, trainloader, validloader, device):
    steps = 0
    print(' \n Initializing training... \n')
    model.to(device)
    for i in range(epochs):
        trainloss = 0
        for inputs, labels in trainloader:
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
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        log_ps = model.forward(inputs)

                        valid_loss += criterion(log_ps, labels).item()
                        ps = torch.exp(log_ps)

                        top_p, top_class = ps.topk(1, dim=1)
                        equal = top_class == labels.view(*top_class.shape)
                        acc += torch.mean(equal.type(torch.FloatTensor)).item()

                print(f"Epoch {i+1}/{epochs}.. "
                      f"Loss: {trainloss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}.. "
                      f"Accuracy: {acc/len(validloader):.3f}")

                trainloss = 0
                model.train()
    print(' \n Successfully Trained ... \n')
    return model
    


def testmodel(model, device, testloader,criterion):
    #Do validation on the test set
    test_loss = 0
    acc = 0
    model.eval()

    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        log_ps = model.forward(inputs)
        
        test_loss += criterion(log_ps, labels).item()
        ps = torch.exp(log_ps)

        top_p, top_class = ps.topk(1, dim=1)
        equal = top_class == labels.view(*top_class.shape)
        acc += torch.mean(equal.type(torch.FloatTensor)).item()

    print(f" Test Loss: {test_loss/len(testloader):.3f}.. "
          f" Accuracy: {acc/len(testloader):.3f}")


    print(' \n Saving the Trained model as Checkpoint.pth \n')


def saved_model(model, train_data, saved, optimizer, arch, classifier):
    # Save the checkpoint
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
        'trained_model': model.name,
        'classifier': model.classifier,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': train_data.class_to_idx,
    }
    torch.save(checkpoint, saved + 'Checkpoint.pth')


    print(' \n Saved successfully as Checkpoint.pth \n')


def main():
    global args

    args = arg_parser()
    #Calling arg_parser

    model = train_model(args.arch)

    model.classifier = model_classifier(model,args.hidden, args.in_features)
    #Using GPU if available
    is_gpu=args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device : {device}")
        
    else:
        device = torch.device("cpu")
        print(f"Device : {device}")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),args.lr)

    trainedmodel = network(model, criterion, optimizer, args.print_every,args.epochs, trainloader, validloader, device)

    testmodel(trainedmodel, device, testloader, criterion)

    save_checkpoint = saved_model(trainedmodel, train_data, args.directory_save, optimizer, args.arch, model.classifier)
#Running the program
    print('\n Done!! \n')
if __name__ == '__main__': main()
