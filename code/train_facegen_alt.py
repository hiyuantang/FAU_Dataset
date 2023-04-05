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
from facegenDataset import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# sampe bash command Windows: 
# python train_facegen.py --seed 1 --dataset_root D:/Datasets/face_gen_single --resume D:/FAU_models/models_r82/checkpoint_epoch49.pth
# sampe bash command Mac: 
# python train_facegen.py --seed 1 --dataset_root /Volumes/Yuan-T7/Datasets/face_gen_single --resume /Volumes/Yuan-T7/FAU_models/models_r82/checkpoint_epoch49.pth

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
parser.add_argument('--seed', default=16, type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', default=50, type=int, help="number of epochs for training")
parser.add_argument('--train_batch_size', default=32, type=int,
                        help="batch size for training")
parser.add_argument('--resume', '-r', default=None, type=str, 
                    help='transfer training by defining the path of stored weights')
parser.add_argument('--test_set', '-t', default=['a', 'e'], type=list, 
                    help='take in a list of skin color scale')
parser.add_argument('--dataset_root', default='D:/Datasets/FAU', 
                    help='the root path of FAU Dataset')
parser.add_argument('--save_interval', default=10, type=int, 
                    help='define the interval of epochs to save model state')
args = parser.parse_args()

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

image_dir = os.path.join(args.dataset_root, 'images')
label_dir = os.path.join(args.dataset_root, 'labels')

if not os.path.isdir('./models_r' + str(args.seed)):
        os.mkdir('./models_r' + str(args.seed))
if not os.path.isdir('./results_r' + str(args.seed)):
        os.mkdir('./results_r' + str(args.seed))

results_path = os.path.join('./results_r'+str(args.seed), 'results.txt')
with open(results_path, 'x') as f:
    pass

num_classes = 10
batch_size = args.train_batch_size
num_epochs = args.epochs

data_transforms = {
        'train': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

print("Initializing Datasets and Dataloaders...")

# load subject folder names from root directory

# subjects --> [a, aw, e, eb]
subjects = np.array([])
for i in next(os.walk(image_dir))[1]:
    # i --> European_Man, European_Woman
    for j in next(os.walk(os.path.join(image_dir, i)))[1]:
        # j --> em1, em2, em3, ..., ew1, ew2, ew3
        subjects = np.append(subjects, j)

test_subjects = np.array(args.test_set)
test_subjects = np.sort(test_subjects)

train_subjects = np.array(subjects, copy=True)
for i in test_subjects:
    index = np.argwhere(train_subjects==i)
    train_subjects = np.delete(train_subjects, index)

print('Train Sets: '+ str(train_subjects))
print('Test Sets: '+ str(test_subjects))

# Train the model
model = VGG_16()
set_parameter_requires_grad(model, feature_extracting=False)
num_ftrs = model.fc8.in_features
model.fc8 = nn.Linear(num_ftrs, num_classes)
if args.resume is None:
    pass
else:
    model.load_state_dict(torch.load(args.resume))

train_dataset = facegenDataset(args.dataset_root, subjects = train_subjects, transform=data_transforms['train'])
test_dataset = facegenDataset(args.dataset_root, subjects = test_subjects, transform=data_transforms['test'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print('Total Number of Train Sets: ' + str(train_dataset.__len__()))
print('Total Number of Test Sets: ' + str(test_dataset.__len__()))
print('---------------Finished---------------')

# Detect if we have a GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Send the model to GPU
model = model.to(device)
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
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

# train model
since = time.time()

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs-1))
    print('-' * 10)

    model.train()  # Set model to training mode
    running_loss = 0.0
    running_pred_label = np.empty((0,20))
        # Iterate over data.
    for images, labels in train_loader:
        inputs = images.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        # track history if only in train
        # Get model outputs and calculate loss
        outputs = model(inputs)
        #labels/tensor([16.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.,  5.]
        loss_batch = criterion(outputs, labels/torch.FloatTensor([16] + [5]*9).to(device))
        loss_batch_mean = loss_batch.mean() * batch_size
        loss_batch_mean.backward()
        optimizer.step()

        running_loss += loss_batch_mean.item()

        outputs = outputs * torch.FloatTensor([16] + [5]*9).to(device)
        running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))

    pred_train = running_pred_label[:,0:10]
    label_train = running_pred_label[:,10:]

    epoch_mses = ((pred_train - label_train)**2).mean(axis=0)
    epoch_maes = (np.abs(pred_train - label_train)).mean(axis=0)

    epoch_pspi_mse = epoch_mses[0]
    epoch_pspi_mae = epoch_maes[0]
    epoch_loss = running_loss / len(train_dataset)
    print('{} loss: {:.4f} PSPI MSE: {:.4f} PSPI MAE: {:.4f}'.format('train', epoch_loss, epoch_pspi_mse, epoch_pspi_mae))

    # add test result
    test_loss, test_mses, test_maes = test_model(model)

    with open(results_path, 'a') as f:
        f.write('train_loss at epoch'+str(epoch)+': '+str(epoch_loss)+'\n')
        f.write('train_mses at epoch'+str(epoch)+': '+str(epoch_mses)+'\n')
        f.write('train_maes at epoch'+str(epoch)+': '+str(epoch_maes)+'\n')
        f.write('test_loss at epoch'+str(epoch)+': '+str(test_loss)+'\n')
        f.write('test_mses at epoch'+str(epoch)+': '+str(test_mses)+'\n')
        f.write('test_maes at epoch'+str(epoch)+': '+str(test_maes)+'\n')
        f.write('\n')
    
    if (epoch+1)%5 == 0:
        torch.save(model.state_dict(), os.path.join('./models_r' + str(args.seed), 'checkpoint_epoch'+str(epoch)+'.pth'))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
