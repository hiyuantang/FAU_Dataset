import sys
sys.path.append('./code')
import argparse
import torch
import torch.nn as nn
import PIL.Image as Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from vgg_face import *
from FAUDataset import *

# dAU7/Delta face change  as \sum dAU7/dfeatureact * Delta featureact/Delta face change 

# sample running command: 
# python derivative_func.py --gender man --seed 66 --activation 5 --au 4

parser = argparse.ArgumentParser(description='d AU# / d face change')
parser.add_argument('--au', default='4', type=str, help='select an au number from [4,6,7,10,12,20,25,26,43]')
parser.add_argument('--gender', default='man', type=str, help='select a gender from [man, woman]')
parser.add_argument('--seed', default='66', type=str, help='select a random seed from [16, 66]')
parser.add_argument('--activation', default=5, type=int, help='select an activation level from [0,1,2,3,4,5]')
parser.add_argument('--featureL', default=0, type=int, help='select a feature layer from 0 to 15')
args = parser.parse_args()

num_classes = 10
batch_size = 32
checkpoint_path = '/Volumes/Yuan-T7/FAU_models/models_r'+args.seed+'/checkpoint_epoch49.pth'
root_path = '/Volumes/Yuan-T7/Datasets/FAU/images'

if args.gender == 'man': 
    gender_path  = os.path.join(root_path, 'European_Man')
    if args.activation == 0:
        image_path_0 = os.path.join(gender_path, 'em5/'+'em5'+'.png')
        image_path_1 = os.path.join(gender_path, 'em6/'+'em6'+'.png')
    else:
        image_path_0 = os.path.join(gender_path, 'em5/'+args.au+'em5_'+args.au+'.'+str(args.activation*2)+'.png')
        image_path_1 = os.path.join(gender_path, 'em6/'+args.au+'em6_'+args.au+'.'+str(args.activation*2)+'.png')
else:
    gender_path  = os.path.join(root_path, 'European_Woman')
    if args.activation == 0:
        image_path_0 = os.path.join(gender_path, 'ew5/'+'ew5'+'.png')
        image_path_1 = os.path.join(gender_path, 'ew6/'+'ew6'+'.png')
    else: 
        image_path_0 = os.path.join(gender_path, 'ew5/'+args.au+'ew5_'+args.au+'.'+str(args.activation*2)+'.png')
        image_path_1 = os.path.join(gender_path, 'ew6/'+args.au+'ew6_'+args.au+'.'+str(args.activation*2)+'.png')


def cal_derivative(image_path):
    data_transforms = {
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ]),
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = VGG_16()
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('mps')), strict=False)

    model.train()

    input_image = Image.open(image_path)
    input_image = input_image.convert('RGB')
    input_image = torch.tensor(np.array(input_image))
    input_image = torch.transpose(input_image, 0, 2).transpose(1, 2)
    input_image = input_image[:, 200:1000, 850:1650]
    input_image = data_transforms['test'](input_image)
    input_image = input_image.view(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
    input_image.requires_grad = True

    model.to('cpu')
    input_image.to('cpu')
    output = model(input_image)
    loss = torch.sum(output)

    loss.backward(retain_graph=True)
    feature_derivative = input_image.grad

    feature_derivative = torch.squeeze(feature_derivative)
    darray = feature_derivative.permute(1, 2, 0).cpu().numpy()
    
    input_image = torch.squeeze(input_image)
    input_image = input_image.permute(1, 2, 0).cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    return darray, input_image, output



darray_light, input_image_light, output_light = cal_derivative(image_path_0)
darray_dark, input_image_dark, output_dark = cal_derivative(image_path_1)

dfeature_light = output_light.sum() / darray_light
dfeature_dark = output_dark.sum() / darray_dark

dfeature = dfeature_light - dfeature_dark
dau = output_light.sum() - output_dark.sum()
dau_dfeature = dau / dfeature

dfeature_dface = dfeature / (input_image_light - input_image_dark)
dfeature_dface[np.isposinf(dfeature_dface)] = 0
dfeature_dface[np.isneginf(dfeature_dface)] = 0

dau_dface = dau_dfeature * dfeature_dface

darray_light = np.sum(darray_light, axis=2)
darray_dark = np.sum(darray_dark, axis=2)
dau_dface = np.sum(dau_dface, axis=2)

normalized_light = np.interp(darray_light, (darray_light.min(), darray_light.max()), (0, 1))
normalized_light = normalized_light - np.mean(normalized_light)
normalized_light[normalized_light < 0] = 0
normalized_dark = np.interp(darray_dark, (darray_dark.min(), darray_dark.max()), (0, 1))
normalized_dark = normalized_dark - np.mean(normalized_dark)
normalized_dark[normalized_dark < 0] = 0
normalized_dau_dface = np.interp(dau_dface, (dau_dface.min(), dau_dface.max()), (0, 1))
normalized_dau_dface = normalized_dau_dface - np.mean(normalized_dau_dface)
normalized_dau_dface[normalized_dau_dface < 0] = 0

fig, axs = plt.subplots(3, 3, figsize=(10, 8))

axs[0,0].imshow(normalized_light, cmap='Greys')
axs[0,0].set_title('dlight_au / dlight_feat')

axs[0,1].imshow(normalized_dark, cmap='Greys')
axs[0,1].set_title('ddark_au / ddark_feat')

im = axs[0,2].imshow(normalized_dau_dface, cmap='Greys')
axs[0,2].set_title('dau / dface')
fig.colorbar(im, ax=axs[0,2])

axs[1,0].imshow(input_image_light+np.tile(normalized_light, (3,1,1)).transpose(1,2,0))
axs[1,1].imshow(input_image_dark+np.tile(normalized_dark, (3,1,1)).transpose(1,2,0))
axs[1,2].imshow(input_image_light+np.tile(normalized_dau_dface, (3,1,1)).transpose(1,2,0))

axs[2,0].imshow(input_image_light)
axs[2,1].imshow(input_image_dark)

y = [0,1,2,3,4,5,6,7,8,9]
axs[2,2].plot(y, output_light.reshape(10,1), marker = 'o', c = '#f7ead0')
axs[2,2].plot(y, output_dark.reshape(10,1), marker = 'o', c = '#3a312a')

for i, ax in enumerate(axs.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 8:
        ax.set_xticks(np.arange(10), ['pspi', '4', '6', '7', '10', '12', '20', '25', '26', '43'])
        ax.set_yticks([0,1])

if args.activation == 0:
    fig.suptitle('model '+'r'+args.seed+'_activation at '+str(args.activation*20)+'%')
else:
    fig.suptitle('model '+'r'+args.seed+'_au'+args.au+'_activation at '+str(args.activation*20)+'%')

plt.show()