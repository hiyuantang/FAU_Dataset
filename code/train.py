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
from FAUDataset import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# sampe bash command: python C:/Users/Yuan/OneDrive/Documents/GitHub/FAU_Dataset/code/train.py --seed 16 --dataset_root C:/Users/Yuan/Datasets/FAU

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

parser = argparse.ArgumentParser(description='FAU Dataset Training')
parser.add_argument('--seed', default=16, type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', default=50, type=int, help="number of epochs for training")
parser.add_argument('--train_batch_size', default=32, type=int,
                        help="batch size for training")
parser.add_argument('--resume', '-r', default=None, type=str, 
                    help='transfer training by defining the path of stored weights')
parser.add_argument('--test_set', '-t', default=[9,10], type=list, 
                    help='take in a list of skin color scale')
parser.add_argument('--dataset_root', default='C:/Users/Yuan/Datasets/FAU', 
                    help='the root path of FAU Dataset')
args = parser.parse_args()

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

image_dir = os.path.join(args.dataset_root, 'images')
label_dir = os.path.join(args.dataset_root, 'labels')

if not os.path.isdir('./models_sf' + str(args.seed)):
        os.mkdir('./models_sf' + str(args.seed))
if not os.path.isdir('./results_sf' + str(args.seed)):
        os.mkdir('./results_sf' + str(args.seed))

num_classes = 10
batch_size = args.train_batch_size
num_epochs = args.epochs

data_transforms = {
        'train': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

print("Initializing Datasets and Dataloaders...")

# load subject folder names from root directory

# subjects --> [em1, em2, em3, ..., ew1, ew2, ew3]
subjects = np.array([])
for i in next(os.walk(image_dir))[1]:
    # i --> European_Man, European_Woman
    for j in next(os.walk(os.path.join(image_dir, i)))[1]:
        # j --> em1, em2, em3, ..., ew1, ew2, ew3
        subjects = np.append(subjects, j)

# general_subjects --> [em, ew, ...]
general_subjects = np.array(list(set([i[0:2] for i in subjects])))

test_subjects = np.array([])
for i in args.test_set:
    append_ary = np.array([j+str(i) for j in general_subjects])
    test_subjects = np.append(test_subjects, append_ary)
    test_subjects = np.sort(test_subjects)

train_subjects = np.array(subjects, copy=True)
for i in test_subjects:
    index = np.argwhere(train_subjects==i)
    train_subjects = np.delete(train_subjects, index)

print('Train Sets: '+ str(train_subjects))
print('Test Sets: '+ str(test_subjects))

# Train the model
model = VGG_16()
if args.resume is None:
    pass
    #model.load_weights()
else:
    model.load_state_dict(torch.load(args.resume))
set_parameter_requires_grad(model, feature_extracting=False)
num_ftrs = model.fc8.in_features
model.fc8 = nn.Linear(num_ftrs, num_classes)
input_size = 224


train_dataset = FAUDataset(args.dataset_root, subjects = train_subjects, transform=data_transforms['train'])
test_dataset = FAUDataset(args.dataset_root, subjects = test_subjects, transform=data_transforms['test'])
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
train_loss_history = []

best_mae = [0, np.Infinity]
best_mse = [0, np.Infinity]

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
        #loss1 = criterion(outputs, labels/torch.FloatTensor([16] + [5]*9).to(device))
        loss_batch = criterion(outputs, labels/torch.FloatTensor([4] + [1]*9).to(device))
        #print(loss_batch.shape) --> torch.Size([32, 10])
        loss_batch = loss_batch.sum(0)
        #print(loss_batch.shape) --> torch.Size([10])
        loss = loss_batch.mean()
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item()

        #outputs = outputs * torch.FloatTensor([16] + [5]*9).to(device)
        outputs = outputs * torch.FloatTensor([4] + [1]*9).to(device)
        running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
    
    pred_test = running_pred_label[:,0:10]
    label_test = running_pred_label[:,10:]

    epoch_mses = ((pred_test - label_test)**2).mean(axis=0)
    epoch_maes = (np.abs(pred_test - label_test)).mean(axis=0)
    # epoch_mse = epoch_mses.mean()
    # epoch_mae = epoch_maes.mean()

    epoch_pspi_mse = epoch_mses[0]
    epoch_pspi_mae = epoch_maes[0]
    epoch_loss = running_loss / len(train_dataset)
    print('{} loss: {:.4f} PSPI MSE: {:.4f} PSPI MAE: {:.4f}'.format('train', epoch_loss, epoch_pspi_mse, epoch_pspi_mae))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
'''
print('Best val MAE: {:4f} at epoch {:0f}'.format(best_mae[1], best_mae[0]))
print('Best val MSE: {:4f} at epoch {:0f}'.format(best_mse[1], best_mse[0]))
f=open('BestEpoch.txt', "a")
f.write('\nBest val MAE: {:4f} at epoch {:0f} \n'.format(best_mae[1], best_mae[0]))
f.write('Best val MSE: {:4f} at epoch {:0f} \n'.format(best_mse[1], best_mse[0]))
f.close()
# load best model weights
model.load_state_dict(best_model_wts)'''
