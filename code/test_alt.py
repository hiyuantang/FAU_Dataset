import sys
sys.path.append('./code')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import numpy as np
import time
import os
import copy
import random
import argparse
from vgg_face import *
from facegenDataset_alt import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# sampe bash command Windows: 
# python test_alt.py --dataset_root D:/Datasets/face_gen --resume D:/FAU_models/checkpoint_epoch_init.pth
# python test_alt.py --dataset_root D:/Datasets/face_gen --resume D:/FAU_models/models_r559/checkpoint_epoch49.pth
# sampe bash command Mac: 
# python test_alt.py --dataset_root /Volumes/Yuan-T7/Datasets/face_gen --resume /Volumes/Yuan-T7/FAU_models/models_r559/checkpoint_epoch49.pth

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

def test_model(model):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_pred_label = np.empty((0,20))
            # Iterate over data.
        for images, labels in test_loader:
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss_batch = criterion(outputs, labels/torch.FloatTensor([16] + [5]*9).to(device))
            loss_batch_mean = loss_batch.mean() * batch_size
            running_loss += loss_batch_mean.item()

            outputs = outputs * torch.FloatTensor([16] + [5]*9).to(device)
            running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
        
        pred_test = running_pred_label[:,0:10]
        label_test = running_pred_label[:,10:]

        mses = ((pred_test - label_test)**2).mean(axis=0)
        maes = (np.abs(pred_test - label_test)).mean(axis=0)

        pspi_mse = mses[0]
        pspi_mae = maes[0]
        loss = running_loss / len(test_dataset)
        print('{} loss: {:.4f} TEST PSPI MSE: {:.4f} TEST PSPI MAE: {:.4f}'.format('test', loss, pspi_mse, pspi_mae)+ '\n')
    return loss, mses, maes

parser = argparse.ArgumentParser(description='FAU Dataset Training & Testing')
parser.add_argument('--fileName', default='test_55_facegen', type=str, help='directory name')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs for training')
parser.add_argument('--train_batch_size', default=32, type=int,
                        help="batch size for training")
parser.add_argument('--resume', '-r', default=None, type=str, 
                    help='transfer training by defining the path of stored weights')
parser.add_argument('--test_set', '-t', default=['a', 'eb', 'e', 'aw'], type=list, 
                    help='take in a list of skin color scale')
parser.add_argument('--dataset_root', default='/Volumes/Yuan-T7/Datasets/FAU', 
                    help='the root path of FAU Dataset')
parser.add_argument('--save_interval', default=10, type=int, 
                    help='define the interval of epochs to save model state')
args = parser.parse_args()

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

image_dir = os.path.join(args.dataset_root, 'images')
label_dir = os.path.join(args.dataset_root, 'labels')

if not os.path.isdir('./results_' + str(args.fileName)):
        os.mkdir('./results_' + str(args.fileName))

results_path = os.path.join('./results_'+str(args.fileName), 'results.txt')
with open(results_path, 'x') as f:
    pass

num_classes = 10
batch_size = args.train_batch_size
num_epochs = args.epochs

data_transforms = {
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }

print("Initializing Datasets and Dataloaders...")

# load subject folder names from root directory

test_subjects = np.array(args.test_set)
test_subjects = np.sort(test_subjects)

print('Test Sets: '+ str(test_subjects))

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Train the model
model = VGG_16()
model = model.to(device)
set_parameter_requires_grad(model, feature_extracting=False)
num_ftrs = model.fc8.in_features
model.fc8 = nn.Linear(num_ftrs, num_classes).to(device)
if args.resume is None:
    pass
else:
    model.load_state_dict(torch.load(args.resume, map_location=device))

test_dataset = facegenDataset(args.dataset_root, subjects = test_subjects, transform=data_transforms['test'])
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Total Number of Test Sets: ' + str(test_dataset.__len__()))
print('---------------Finished---------------')

params_to_update = model.parameters()
print("Params to learn:")
feature_extract = False
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
last_layer = list(model.children())[-1]
try:
    last_layer = last_layer[-1]
except:
    last_layer = last_layer
ignored_params = list(map(id, last_layer.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params,
                model.parameters())

optimizer = optim.Adam([
            {'params': base_params},
            {'params': last_layer.parameters(), 'lr': 1e-4}
        ], lr=1e-5, weight_decay=5e-4)

# Setup the loss fxn
criterion = nn.MSELoss(reduction='none')



# add test result
test_loss, test_mses, test_maes = test_model(model)

with open(results_path, 'a') as f:
    f.write('test_loss'+': '+str(test_loss)+'\n')
    f.write('test_mses'+': '+str(test_mses)+'\n')
    f.write('test_maes'+': '+str(test_maes)+'\n')
    f.write('\n')

print('Training complete')
