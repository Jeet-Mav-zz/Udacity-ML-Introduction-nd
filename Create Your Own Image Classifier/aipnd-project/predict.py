import argparse, json
from collections import OrderedDict
import numpy as np
import torch
from torch import nn, optim
from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from math import ceil



def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='./Checkpoint.pth', type =str, action="store", help = 'Load Checkpoint')
    parser.add_argument('--image_path', default = 'flowers/test/1/image_06743.jpg',
                        type =str, action="store",help = 'Path of Image')
    parser.add_argument('--gpu', default="gpu", action="store", help='USE GPU')
    parser.add_argument('--topk', default="5", action="store", type = int, help='Top No of Matches')
    parser.add_argument('--category', default='cat_to_name.json', type=str, action="store", help='Map Flower names to its category')
    
    args = parser.parse_args()
    return args


def rebuild(filepath):
    checkpoint = torch.load(filepath)
    model = model = getattr(torchvision.models, checkpoint['trained_model'])(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.optimizer = checkpoint['optimizer']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    print('\n Successfully Loaded Model \n')
    
    print(model)
    
    return model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    #Process a PIL image for use in a PyTorch model
    
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    im = Image.open(image_path)
    image_data = transform(im)
    
    return image_data


def predict(image_path, model,device, topk, category):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    model.to(device)
    
    # Predict the class from an image file
    np_image = image_path
    np_image = torch.from_numpy(np.expand_dims(np_image, axis=0)).float()
    image = np_image.to(device)
    
    with torch.no_grad():
        output = model(image)
    
    prob, label = output.topk(topk)
    prob = np.array(prob.exp().data)[0]
    labels = np.array(label)[0]
   
    # Convert to class
    idx_to_class = {x:y for y, x in model.class_to_idx.items()}
    labels = [idx_to_class[i] for i in labels]
    flowers = [category[i] for i in labels]
        
    return prob, labels, flowers


def main():
    global args

    args = arg_parser()
    #Calling arg_parser
    
    with open(args.category, 'r') as f:
        cat_to_name = json.load(f)
        
    model = rebuild(args.checkpoint)
    
    image_path = args.image_path
    image_tensor = process_image(image_path)
    
    is_gpu=args.gpu

    use_cuda = torch.cuda.is_available()
    device = torch.device("cpu")
    if is_gpu and use_cuda:
        device = torch.device("cuda:0")
        print(f"Device : {device}")
        
    else:
        device = torch.device("cpu")
        print(f"Device : {device}")
        
        
    prob, labels, flowers = predict(image_tensor, model, device, args.topk, cat_to_name)
    
    for i, j in enumerate(zip(prob,flowers)):
        print (" {}, Probability : {} % ".format(j[1].upper(), ceil(j[0]*100)))
        
#Running the progran
    print('\n Done!! \n')
if __name__ == '__main__': main()    
    