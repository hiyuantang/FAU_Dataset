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
# python derivative_func_UNBC.py --seed init --au 6

parser = argparse.ArgumentParser(description='d AU# / d face change')
parser.add_argument('--au', default='4', type=str, help='select an au number from [4,6,7,10,12,20,25,26,43]')
parser.add_argument('--seed', default='init', type=str, help='select a random seed from [init, 16, 66]')
args = parser.parse_args()


num_classes = 10
batch_size = 32
if args.seed == 'init':
    checkpoint_path = '/Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth'
else:
    checkpoint_path = '/Volumes/Yuan-T7/FAU_models/models_r'+args.seed+'/checkpoint_epoch49.pth'
root_path = '/Volumes/Yuan-T7/Datasets/FAU/images'


image_path_0 = '/Volumes/Yuan-T7/Datasets/UNBCMcMaster/Images/080-bn080/bn080t1afaff/bn080t1afaff200.png'
image_path_1 = '/Volumes/Yuan-T7/Datasets/UNBCMcMaster/Images/042-ll042/ll042t1afaff/ll042t1afaff029.png'

au_path_0 = '/Volumes/Yuan-T7/Datasets/UNBCMcMaster/Frame_Labels/FACS/080-bn080/bn080t1afaff/bn080t1afaff200_facs'
au_path_1 = '/Volumes/Yuan-T7/Datasets/UNBCMcMaster/Frame_Labels/FACS/042-ll042/ll042t1afaff/ll042t1afaff029_facs'

# choose index from 2 to inf
def array_repeat(x, index):
    x = np.repeat(x, index, axis = 0)
    x = np.repeat(x, index, axis = 1)
    return x

def read_au(path, dtype = 'txt'):
    if dtype == 'txt':
        f=open(path + '.txt', "r")
        scorestr = f.readlines()
        f.close()
        scorestr = [x.strip() for x in scorestr]
        au = np.zeros((64,))
        for line in scorestr:
            words = [x.strip() for x in line.split(' ') if x]
            aunumberstr = words[0]
            auintensitystr = words[1]
            aunumber = float(aunumberstr[0:aunumberstr.find('e')]) * (10** int(aunumberstr[aunumberstr.find('+')+1:]))
            auintensity = float(auintensitystr[0:auintensitystr.find('e')]) * (10** int(auintensitystr[auintensitystr.find('+')+1:]))
            au[int(aunumber)-1] = auintensity
        au = np.array(au[[3,5,6,9,11,19,24,25,42]])

    elif dtype == 'npy':
        au = np.load(path+'.npy')
    elif dtype == 'npz':
        au = np.load(path+'.npz')['output']

    return au

def cal_derivative(image_path, cut, img_num):
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
        if img_num == 0:
            input_image = input_image[:, 50:220, 25:195]
        elif img_num == 1:
            input_image = input_image[:, 50:200, 115:265]
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
    darray_light, input_image_light, inter_light, output_light = cal_derivative(image_path_0, cut, 0)
    darray_dark, input_image_dark, inter_dark, output_dark = cal_derivative(image_path_1, cut, 1)

    dfeature = inter_light - inter_dark

    dau_dface_light = torch.squeeze(darray_light * dfeature)
    dau_dface_dark = torch.squeeze(darray_dark * dfeature)

    dau_dface_light = np.sum(dau_dface_light.numpy(), axis=0)/dau_dface_light.shape[0]
    dau_dface_dark= np.sum(dau_dface_dark.numpy(), axis=0)/dau_dface_dark.shape[0]

    dau_dface_light = array_repeat(dau_dface_light, 2**(cut+1))
    dau_dface_dark = array_repeat(dau_dface_dark, 2**(cut+1))

    normalized_light = np.interp(dau_dface_light, (dau_dface_light.min(), dau_dface_light.max()), (0, 1))
    normalized_light = normalized_light - np.mean(normalized_light)
    #normalized_light[normalized_light < 0] = 0
    normalized_dark = np.interp(dau_dface_dark, (dau_dface_dark.min(), dau_dface_dark.max()), (0, 1))
    normalized_dark = normalized_dark - np.mean(normalized_dark)
    #normalized_dark[normalized_dark < 0] = 0

    axs[0,cut].imshow(dau_dface_light, cmap='Greys')
    axs[0,cut].set_title('hidden_state'+str(cut))

    #axs[0,1].imshow(normalized_dark, cmap='Greys')
    #axs[0,1].set_title('ddark_au / dface')
    fig.colorbar(axs[0,cut].imshow(dau_dface_light, cmap='Greys'), ax=axs[0,cut])

    axs[1,cut].imshow(input_image_light+np.tile(normalized_light, (3,1,1)).transpose(1,2,0))
    #axs[1,1].imshow(input_image_dark+np.tile(normalized_dark, (3,1,1)).transpose(1,2,0))

    if cut == 0:
        axs[2,0].imshow(input_image_light)
        axs[2,2].imshow(input_image_dark)
        #axs[2,1].imshow(input_image_dark)
    else:
        pass

##########################
au_0 = read_au(au_path_0)
au_1 = read_au(au_path_1)
pspi_score_0 = au_0[0]+max(au_0[1], au_0[2])+max(0, au_0[3])+au_0[8]
pspi_score_1 = au_1[0]+max(au_1[1], au_1[2])+max(0, au_1[3])+au_1[8]
pspi_au_0 = np.insert(au_0, 0, pspi_score_0)/([16] + [5]*9)
pspi_au_1 = np.insert(au_1, 0, pspi_score_1)/([16] + [5]*9)

y = [0,1,2,3,4,5,6,7,8,9]
axs[2,1].plot(y, output_light.reshape(10,1), marker = 'o', c = '#f7ead0')
axs[2,1].plot(y, pspi_au_0, marker = '_', c = '#f7ead0')
axs[2,3].plot(y, output_dark.reshape(10,1), marker = 'o', c = '#3a312a')
axs[2,3].plot(y, pspi_au_1, marker = '_', c = '#3a312a')


#axs[2,2].axis('off')
#axs[2,3].axis('off')
axs[2,4].axis('off')
##########################

for i, ax in enumerate(axs.flat):
    ax.set_xticks([])
    ax.set_yticks([])
    if i in [11, 13] :
        ax.set_xticks(np.arange(10), ['pspi', '4', '6', '7', '10', '12', '20', '25', '26', '43'])
        ax.set_yticks([0,1])


fig.suptitle('model '+args.seed+' | au'+args.au)

plt.show()