import sys
sys.path.append('./code')
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import numpy as np
import time
import os
import random
import argparse
from vgg_face import *
import matplotlib.pyplot as plt
from facegenDataset import *
from FAUDataset import *
plt.switch_backend('agg')

# sampe bash command Windows: 
# python train.py --dataset_root D:/Datasets/FAU --resume D:/FAU_models/checkpoint_epoch_init.pth --seed 32 --mode 0
# sampe bash command Mac: 
# python train.py --dataset_root /Volumes/Yuan-T7/Datasets/FAU --resume /Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth --seed 10 --mode 0

parser = argparse.ArgumentParser(description='FAU Dataset Training & Testing')
parser.add_argument('--seed', default=16, type=int, help='seed for initializing training. ')
parser.add_argument('--epochs', default=70, type=int, help="number of epochs for training")
parser.add_argument('--train_batch_size', default=32, type=int,
                        help="batch size for training")
parser.add_argument('--resume', '-r', default=None, type=str, 
                    help='transfer training by defining the path of stored weights')
parser.add_argument('--train_set', default=['em6','em7','em8','em9','em10','ew6','ew7','ew8','ew9','ew10'], type=list, 
                    help='take in a list of skin color scale')
parser.add_argument('--test_set', default=['em1','em2','em3','em4','em5','ew1','ew2','ew3','ew4','ew5'], type=list, 
                    help='take in a list of skin color scale')
#parser.add_argument('--train_set', default=['aw'], type=list, 
#                    help='take in a list of skin color scale')
#parser.add_argument('--test_set', default=['eb'], type=list, 
#                    help='take in a list of skin color scale')
parser.add_argument('--dataset_root', default='D:/Datasets/FAU', 
                    help='the root path of dataset')
parser.add_argument('--save_interval', default=70, type=int, 
                    help='define the interval of epochs to save model state')
parser.add_argument('--mode', default=0, type=int, 
                    help='define the data pre-transformation mode: mode 0 is non-transformed')
args = parser.parse_args()

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

def comma_array(tensor_array):
    return (', '.join(str(x) for x in tensor_array))

def test_model(model, test_dataset, test_loader, device, criterion, batch_size):
    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        running_pred_label = np.empty((0,22))
            # Iterate over data.
        for images, labels in test_loader:
            inputs = images.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss_batch = criterion(outputs, labels/torch.FloatTensor([16] + [5]*10).to(device))
            loss_batch_mean = loss_batch.mean() * batch_size
            running_loss += loss_batch_mean.item()

            outputs = outputs * torch.FloatTensor([16] + [5]*10).to(device)
            running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))
        
        pred_test = running_pred_label[:,0:11]
        label_test = running_pred_label[:,11:]
        mses, maes, mses_single_au, maes_single_au, pspi_mse, pspi_mae, pred_avg, pred_avg_diff = epoch_losses(pred_test, label_test)

        loss = running_loss / len(test_dataset)
        print('{} loss: {:.4f} TEST PSPI MSE: {:.4f} TEST PSPI MAE: {:.4f}'.format('test', loss, pspi_mse, pspi_mae)+ '\n')
    return loss, mses, maes, mses_single_au, maes_single_au, pred_avg, pred_avg_diff

def epoch_losses(predictions, labels):
    mses = ((predictions - labels)**2).mean(axis=0)
    maes = (np.abs(predictions - labels)).mean(axis=0)

    pred_transformed = np.where(labels != 0, predictions, 0)
    mses_single_au = ((pred_transformed - labels)**2).mean(axis=0)
    maes_single_au = (np.abs(pred_transformed - labels)).mean(axis=0)
    pred_avg = pred_transformed.mean(axis=0)
    pred_avg_diff = (pred_transformed-labels).mean(axis=0)

    pspi_mse = mses[0]
    pspi_mae = maes[0]
    return mses, maes, mses_single_au, maes_single_au, pspi_mse, pspi_mae, pred_avg, pred_avg_diff

def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    image_dir = os.path.join(args.dataset_root, 'images')
    label_dir = os.path.join(args.dataset_root, 'labels')

    if not os.path.isdir('./models_r' + str(args.seed)):
            os.mkdir('./models_r' + str(args.seed))
    if not os.path.isdir('./results_r' + str(args.seed)):
            os.mkdir('./results_r' + str(args.seed))

    results_path = os.path.join('./results_r'+str(args.seed), 'results.txt')
    with open(results_path, 'x') as f:
        pass

    num_classes = 11
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

    train_subjects = args.train_set
    test_subjects = args.test_set
    print('Train Sets: '+ str(train_subjects))
    print('Test Sets: '+ str(test_subjects))

    # define model
    model = VGG_16().to(device)
    set_parameter_requires_grad(model, feature_extracting=False)
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, num_classes).to(device)
    if args.resume is None:
        pass
    else:
        checkp = torch.load(args.resume, map_location=device)
        ignored_layer_keys = ['fc8.weight', 'fc8.bias']
        filtered_dict = {k: v for k, v in checkp.items() if k not in ignored_layer_keys}
        model_dict = model.state_dict()
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

    # load data
    if 'FAU' in args.dataset_root:
        train_dataset = FAUDataset(args.dataset_root, subjects = train_subjects, mode=args.mode, transform=data_transforms['train'])
        test_dataset = FAUDataset(args.dataset_root, subjects = test_subjects, mode=args.mode, transform=data_transforms['test'])
    elif 'facegen' in args.dataset_root:
        train_dataset = facegenDataset(args.dataset_root, subjects = train_subjects, mode=args.mode, transform=data_transforms['train'])
        test_dataset = facegenDataset(args.dataset_root, subjects = test_subjects, mode=args.mode, transform=data_transforms['test'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print('Total Number of Train Sets: ' + str(train_dataset.__len__()))
    print('Total Number of Test Sets: ' + str(test_dataset.__len__()))
    print('---------------Finished---------------')

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

    # Train model
    since = time.time()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        model.train()  # Set model to training mode
        running_loss = 0.0
        running_pred_label = np.empty((0,22))
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
            loss_batch = criterion(outputs, labels/torch.FloatTensor([16] + [5]*10).to(device))
            loss_batch_mean = loss_batch.mean() * batch_size
            loss_batch_mean.backward()
            optimizer.step()

            running_loss += loss_batch_mean.item()

            outputs = outputs * torch.FloatTensor([16] + [5]*10).to(device)
            running_pred_label = np.concatenate((running_pred_label, np.concatenate([outputs.data.cpu().numpy(), labels.data.cpu().numpy()],axis=1)))

        pred_train = running_pred_label[:,0:11]
        label_train = running_pred_label[:,11:]

        train_mses, train_maes, train_mses_single_au, train_maes_single_au, train_pspi_mse, train_pspi_mae, train_pred_avg, train_pred_avg_diff = epoch_losses(pred_train, label_train)

        epoch_loss = running_loss / len(train_dataset)
        print('{} loss: {:.4f} PSPI MSE: {:.4f} PSPI MAE: {:.4f}'.format('train', epoch_loss, train_pspi_mse, train_pspi_mae))

        # add test result
        test_loss, test_mses, test_maes, test_mses_single_au, test_maes_single_au, test_pred_avg, test_pred_avg_diff = test_model(model, test_dataset, test_loader, device, criterion, batch_size)

        with open(results_path, 'a') as f:
            if epoch == 0:
                f.write('Train Sets: ' + str(args.train_set) + ' | Test Sets: ' + str(args.test_set)+'\n')
                f.write('Mode: ' + str(args.mode)+'\n')
            f.write('train_loss at epoch'+str(epoch)+': '+str(epoch_loss)+'\n')
            f.write('train_mses at epoch'+str(epoch)+': '+comma_array(train_mses)+'\n')
            f.write('train_maes at epoch'+str(epoch)+': '+comma_array(train_maes)+'\n')
            f.write('train_mses_single at epoch'+str(epoch)+': '+comma_array(train_mses_single_au)+'\n')
            f.write('train_maes_single at epoch'+str(epoch)+': '+comma_array(train_maes_single_au)+'\n')
            f.write('train_pred_single_avg at epoch'+str(epoch)+': '+comma_array(train_pred_avg)+'\n')
            f.write('train_pred_single_avg_diff at epoch'+str(epoch)+': '+comma_array(train_pred_avg_diff)+'\n')
            f.write('\n')
            f.write('test_loss at epoch'+str(epoch)+': '+str(test_loss)+'\n')
            f.write('test_mses at epoch'+str(epoch)+': '+comma_array(test_mses)+'\n')
            f.write('test_maes at epoch'+str(epoch)+': '+comma_array(test_maes)+'\n')
            f.write('test_mses_single at epoch'+str(epoch)+': '+comma_array(test_mses_single_au)+'\n')
            f.write('test_maes_single at epoch'+str(epoch)+': '+comma_array(test_maes_single_au)+'\n')
            f.write('test_pred_single_avg at epoch'+str(epoch)+': '+comma_array(test_pred_avg)+'\n')
            f.write('test_pred_single_avg_diff at epoch'+str(epoch)+': '+comma_array(test_pred_avg_diff)+'\n')
            f.write('\n')
            f.write('\n')
            f.write('\n')
        
        if (epoch+1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join('./models_r' + str(args.seed), 'checkpoint_epoch'+str(epoch)+'.pth'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == '__main__':
    main()