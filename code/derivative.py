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

parser = argparse.ArgumentParser(description='derivative / feature')
parser.add_argument('--au', default='4', type=str, help='select an au number')
parser.add_argument('--gender', default='man', type=str, help='select a gender')
parser.add_argument('--seed', default='66', type=str, help='select a random seed')
args = parser.parse_args()

num_classes = 10
batch_size = 32
checkpoint_path = '/Volumes/Yuan-T7/FAU_models/models_r'+args.seed+'/checkpoint_epoch49.pth'
root_path = '/Volumes/Yuan-T7/Datasets/FAU/images'

if args.gender == 'man':    
    gender_path  = os.path.join(root_path, 'European_Man')
    image_path_0 = os.path.join(gender_path, 'em1/'+args.au+'em1_'+args.au+'.10.png')
    image_path_1 = os.path.join(gender_path, 'em10/'+args.au+'em10_'+args.au+'.10.png')
else:
    gender_path  = os.path.join(root_path, 'European_Woman')
    image_path_0 = os.path.join(gender_path, 'ew1/'+args.au+'ew1_'+args.au+'.10.png')
    image_path_1 = os.path.join(gender_path, 'ew10/'+args.au+'ew10_'+args.au+'.10.png')


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
    array = feature_derivative.permute(1, 2, 0).cpu().numpy()
    
    input_image = torch.squeeze(input_image)
    input_image = input_image.permute(1, 2, 0).cpu().detach().numpy()
    output = output.cpu().detach().numpy()

    return array, input_image, output



darray_light, input_image_light, output_light = cal_derivative(image_path_0)
darray_dark, input_image_dark, output_dark = cal_derivative(image_path_1)

darray_light = np.sum(darray_light, axis=2)
darray_dark = np.sum(darray_dark, axis=2)

diff_array = darray_light - darray_dark

normalized_light = np.interp(darray_light, (darray_light.min(), darray_light.max()), (0, 1))
normalized_light = normalized_light - np.mean(normalized_light)
normalized_light[normalized_light < 0] = 0
normalized_dark = np.interp(darray_dark, (darray_dark.min(), darray_dark.max()), (0, 1))
normalized_dark = normalized_dark - np.mean(normalized_dark)
normalized_dark[normalized_dark < 0] = 0
normalized_diff = np.interp(diff_array, (diff_array.min(), diff_array.max()), (0, 1))
normalized_diff = normalized_diff - np.mean(normalized_diff)
normalized_diff[normalized_diff < 0] = 0

fig, axs = plt.subplots(2, 3, figsize=(10, 6))

axs[0,0].imshow(normalized_light, cmap='Greys')
axs[0,0].set_title('light')

axs[0,1].imshow(normalized_dark, cmap='Greys')
axs[0,1].set_title('dark')

im = axs[0,2].imshow(normalized_diff, cmap='Greys')
axs[0,2].set_title('difference (light - dark)')
fig.colorbar(im, ax=axs[0,2])

axs[1,0].imshow(input_image_light)
axs[1,1].imshow(input_image_dark)

y = [0,1,2,3,4,5,6,7,8,9]
axs[1,2].plot(y, output_light.reshape(10,1), marker = 'o', c = '#f7ead0')
axs[1,2].plot(y, output_dark.reshape(10,1), marker = 'o', c = '#3a312a')

for i, ax in enumerate(axs.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 5:
        ax.set_xticks(np.arange(10), ['pspi', '4', '6', '7', '10', '12', '20', '25', '26', '43'])
        ax.set_yticks([0,1])

fig.suptitle('r'+args.seed+'_au'+args.au+'_max activation')

plt.show()