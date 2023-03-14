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
# python derivative_func_facegen.py --race african --seed init --activation 5 --au 4

parser = argparse.ArgumentParser(description='d AU# / d face change')
parser.add_argument('--au', default='4', type=str, help='select an au number from [4,6,7,10,12,20,25,26,43]')
parser.add_argument('--race', default='african', type=str, help='select a gender from [african, european]')
parser.add_argument('--seed', default='66', type=str, help='select a random seed from [init, 16, 66]')
parser.add_argument('--activation', default=5, type=int, help='select an activation level from [1,2,3,4,5]')
args = parser.parse_args()


num_classes = 10
batch_size = 32
if args.seed == 'init':
    checkpoint_path = '/Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth'
else:
    checkpoint_path = '/Volumes/Yuan-T7/FAU_models/models_r'+args.seed+'/checkpoint_epoch49.pth'
root_path = '/Volumes/Yuan-T7/Datasets/face_gen/images'

if args.race == 'african': 
    race_path  = os.path.join(root_path, 'African')
    image_path_0 = os.path.join(race_path, 'aw/'+args.au+'aw_'+args.au+'.'+str(args.activation*2)+'.png')
    image_path_1 = os.path.join(race_path, 'a/'+args.au+'a_'+args.au+'.'+str(args.activation*2)+'.png')
else:
    race_path  = os.path.join(root_path, 'European')
    image_path_0 = os.path.join(race_path, 'e/'+args.au+'e_'+args.au+'.'+str(args.activation*2)+'.png')
    image_path_1 = os.path.join(race_path, 'eb/'+args.au+'eb_'+args.au+'.'+str(args.activation*2)+'.png')

# choose index from 2 to inf
def array_repeat(x, index):
    x = np.repeat(x, index, axis = 0)
    x = np.repeat(x, index, axis = 1)
    return x

def cal_derivative(image_path, cut):
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
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('mps')))
    model.to('cpu')

    model.eval()
    with torch.no_grad():
        input_image = Image.open(image_path)
        input_image = input_image.convert('RGB')
        input_image = torch.tensor(np.array(input_image)).to('cpu')
        input_image = torch.transpose(input_image, 0, 2).transpose(1, 2)
        input_image = input_image[:, 100:700, 50:650]
        input_image = data_transforms['test'](input_image)
        input_image = input_image.view(1, input_image.shape[0], input_image.shape[1], input_image.shape[2])
        input_inter = model.forward_thoughts(input_image, cut)
    
    model.train()
    input_inter.requires_grad = True

    if cut == 0:
        output = model.forward_0(input_inter)
    elif cut == 1:
        output = model.forward_1(input_inter)
    elif cut == 2:
        output = model.forward_2(input_inter)
    elif cut == 3:
        output = model.forward_3(input_inter)
    elif cut == 4:
        output = model.forward_4(input_inter)
    au_list = ['pspi', '4', '6', '7', '10', '12', '20', '25', '26', '43']
    loss = output[0, au_list.index(args.au)]
    loss.backward(retain_graph=True)
    feature_derivative = input_inter.grad

    #feature_derivative = torch.squeeze(feature_derivative)
    #darray = feature_derivative.permute(1, 2, 0).cpu().numpy()
    
    input_image = torch.squeeze(input_image)
    input_image = input_image.permute(1, 2, 0).cpu().detach().numpy()
    input_inter = input_inter.cpu().detach().numpy()

    output = output.cpu().detach().numpy()

    return feature_derivative, input_image, input_inter, output

fig, axs = plt.subplots(3, 5, figsize=(13, 8))
for cut in range(5):
    darray_light, input_image_light, inter_light, output_light = cal_derivative(image_path_0, cut)
    darray_dark, input_image_dark, inter_dark, output_dark = cal_derivative(image_path_1, cut)

    dfeature = inter_light - inter_dark

    dau_dface_light = torch.squeeze(darray_light * dfeature)
    dau_dface_dark = torch.squeeze(darray_dark * dfeature)

    dau_dface_light = np.sum(dau_dface_dark.numpy(), axis=0)
    dau_dface_dark= np.sum(dau_dface_dark.numpy(), axis=0)


    normalized_light = np.interp(dau_dface_light, (dau_dface_light.min(), dau_dface_light.max()), (0, 1))
    normalized_light = normalized_light - np.mean(normalized_light)
    normalized_light[normalized_light < 0] = 0
    normalized_light = array_repeat(normalized_light, 2**(cut+1))
    normalized_dark = np.interp(dau_dface_dark, (dau_dface_dark.min(), dau_dface_dark.max()), (0, 1))
    normalized_dark = normalized_dark - np.mean(normalized_dark)
    normalized_dark[normalized_dark < 0] = 0
    normalized_dark = array_repeat(normalized_dark, 2**(cut+1))

    axs[0,cut].imshow(normalized_light, cmap='Greys')
    axs[0,cut].set_title('hidden_state'+str(cut))

    #axs[0,1].imshow(normalized_dark, cmap='Greys')
    #axs[0,1].set_title('ddark_au / dface')

    axs[1,cut].imshow(input_image_light+np.tile(normalized_light, (3,1,1)).transpose(1,2,0))
    #axs[1,1].imshow(input_image_dark+np.tile(normalized_dark, (3,1,1)).transpose(1,2,0))

    if cut == 0:
        axs[2,cut].imshow(input_image_light)
        #axs[2,1].imshow(input_image_dark)
    else:
        pass

##########################
y = [0,1,2,3,4,5,6,7,8,9]
axs[2,1].plot(y, output_light.reshape(10,1), marker = 'o', c = '#f7ead0')
axs[2,1].plot(y, output_dark.reshape(10,1), marker = 'o', c = '#3a312a')

axs[2,2].axis('off')
axs[2,3].axis('off')
axs[2,4].axis('off')
##########################

for i, ax in enumerate(axs.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    if i == 11:
        ax.set_xticks(np.arange(10), ['pspi', '4', '6', '7', '10', '12', '20', '25', '26', '43'])
        ax.set_yticks([0,1])

if args.activation == 0:
    fig.suptitle('model '+args.seed+' | activation at '+str(args.activation*20)+'%')
else:
    fig.suptitle('model '+args.seed+' | au'+args.au+' | activation at '+str(args.activation*20)+'%')

plt.show()