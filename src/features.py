import torch
import torch.nn as nn
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#from plotly.subplots import make_subplots
import torchvision.transforms as transforms
#import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from os.path import join, basename, exists
from torchvision.models import resnet50, ResNet50_Weights, convnext_tiny, ConvNeXt_Tiny_Weights, vit_b_16, ViT_B_16_Weights, swin_t, Swin_T_Weights
import torch.optim as optim
#from torch.optim.lr_scheduler import StepLR
import random
from os import listdir, environ
#from sklearn.model_selection import train_test_split
#import glob
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from PIL import Image
#import json
#import openTSNE
#import sklearn.manifold
#import time
from aux import create_dir

def seed_everything(seed):
    random.seed(seed)
    environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

def freeze_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

class ILTDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, img_path

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_model():
    model = convnext_tiny(weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1)    
    return model

def register_hooks(model):
    model.features[3][2].block[5].register_forward_hook(get_activation('layer1'))
    model.features[5][8].block[5].register_forward_hook(get_activation('layer2'))
    model.features[7][2].block[5].register_forward_hook(get_activation('layer3'))
    model.classifier[0].register_forward_hook(get_activation('layer4'))


# Input:
  # (1) images_folder: a string with the path to the folder with the images
  # (2) project_name: a string used to name the saved the files (optional, if
  #                   not provided, it uses the basename of the images_folder)
  # (3) weights_path: a string with the path to the weights to load (optional,
  #                   if not provided, loads weights from the ImageNet)
# Output:
def compute_features(images_folder, batch_start, batch_end, weights_path = ''):
    print('Computing features.')
    batch_size = 32
    device = 'cuda'

    lr = 3e-5
#    gamma = 0.7
    seed = 0

    output_path = join(images_folder, 'predictions')
    create_dir(output_path)
    seed_everything(seed)
    test_transform = get_transforms()

    model = get_model()
    freeze_bn(model)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    dev = torch.cuda.current_device()

    if weights_path != '':
        checkpoint = torch.load(weights_path, map_location = lambda storage, loc: storage.cuda(dev))

        model.load_state_dict(checkpoint['model'])
        model.to(device)  ##important to do BEFORE loading the optimizer
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
    else:
        model.to(device)  ##important to do BEFORE loading the optimizer
        
    register_hooks(model)

    for i in range(batch_start, batch_end + 1):
        batch_id = 'batch_{:04d}'.format(i)
        activation = {}
            
        test_list = [join(images_folder, l) for l in listdir(join(images_folder, batch_id))]
        test_data = ILTDataset(test_list, transform=test_transform)
        test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)

        images_path = []
        predictions = []
        features = None

        with torch.no_grad():
            for data, paths in tqdm(test_loader):
                data = data.to(device)
                
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    output = model(data)
                    if features is None:
                        features = torch.amax(activation['layer4'], (2, 3))
                    else:
                        aux = torch.amax(activation['layer4'], (2, 3))
                        features = torch.vstack((features, aux))
                    
                paths = list(paths)
                preds = output.argmax(dim=1)
                preds_list = []
                for i in range(preds.shape[0]):
                    preds_list.append(preds[i].item())
                    paths[i] = basename(paths[i])
                predictions.extend(preds_list)
                images_path.extend(paths)
        
        features = features.cpu().detach().numpy()

    return features, paths
    
    arr_files = np.array(images_path).reshape(len(images_path), -1)
    arr = np.hstack([arr_files, features])
    cs = ['names']
    for i in range(features.shape[1]):
        cs.append('f_' + str(i+1))
    df_features = pd.DataFrame(arr, columns = cs)
    df_features

#    df_features.to_csv('features.csv', index=None)

    cur = timer()
    time_diff = cur - start
