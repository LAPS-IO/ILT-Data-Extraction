import os
import random
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
from aux import defaults

activation = {}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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
        img = PIL.Image.open(img_path).convert('RGB')
        img_transformed = self.transform(img)

        return img_transformed, img_path

def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def get_model(load=False, num_classes=1000):
    model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    if load:
        model.classifier[2] = nn.Linear(768, num_classes)
    return model

def register_hooks(model,):
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
def compute_features(images_folder, batch_id, model, weights_path, labels_dict = None):
    global activation

    batch_size = 128
    device = 'cuda'

    lr = 3e-5
    # gamma = 0.7
    seed = 0
    torch.backends.cudnn.benchmark = True

    seed_everything(seed)
    test_transform = get_transforms()

    freeze_bn(model)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    use_amp = True
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    dev = torch.cuda.current_device()

    model.to(device) # important to do BEFORE loading the optimizer
    if weights_path != '':
        checkpoint = torch.load(weights_path, weights_only=True,
                                map_location=lambda storage, loc: storage.cuda(dev))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])

    register_hooks(model)

    activation = {}

    inner_folder = os.path.join(images_folder, batch_id, defaults['inner_folder'])
    file_list = os.listdir(inner_folder)
    test_list = [os.path.join(inner_folder, file) for file in file_list]
    test_data = ILTDataset(test_list, transform=test_transform)
    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=1)

    path_images, predictions, confs = [], [], []
    features = None

    with torch.no_grad():
        for data, paths in test_loader:
            data = data.to(device)
            with torch.amp.autocast(device, dtype=torch.float16):
                output = model(data)
                probabilities = F.softmax(output, dim=1) 
                aux = torch.amax(activation['layer4'], (2, 3))
                if features is None:
                    features = aux
                else:
                    features = torch.vstack((features, aux))

            paths = list(paths)
            paths = [os.path.basename(path) for path in paths]
            _, preds = torch.max(probabilities, dim=1)
            top3_confidences, top3_classes = torch.topk(torch.softmax(output, dim=1), k=3, dim=1)

            preds_list = []
            confs_list = []
            for i in range(preds.shape[0]):
                preds_list.append(preds[i].item())
                conf = {}
                for j in range(3):
                    class_id = top3_classes[i, j].item()
                    if labels_dict is not None:
                        class_id = list(labels_dict.keys())[class_id]
                    confidence = top3_confidences[i, j].item()
                    conf[f'top{j+1}'] = (class_id, confidence)
                confs_list.append(conf)
                paths[i] = os.path.basename(paths[i])
            predictions.extend(preds_list)
            confs.extend(confs_list)
            path_images.extend(paths)
        features = features.cpu().detach().numpy()

    return features, path_images, predictions, confs
