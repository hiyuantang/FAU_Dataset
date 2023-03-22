import os
import random
import numpy as np
import sys
#sys.path.append('./shoulder_pain_detection')
import McMasterDataset
from McMasterDataset import *
from vgg_face import *
from torchvision import datasets, models, transforms
import torch

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "face1_vgg":
        """ vgg-vd-16 trained on vggface
        """
        model_ft = VGG_16()
        model_ft.load_weights()
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc8.in_features
        model_ft.fc8 = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
image_dir = '/home/AD/yutang/UNBCMcMaster_cropped/Images0.3'
label_dir = '/home/AD/yutang/UNBCMcMaster'

subjects = []
for d in next(os.walk(image_dir))[1]:
    subjects.append(d[:3])
subjects = sorted(subjects)
random.shuffle(subjects)

fold_size = 5
folds = []
for i in range(5):
    folds += [subjects[i*fold_size: (i+1)*fold_size]]

data_transforms = {
        'train': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'val': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
        'test': transforms.Compose([
            # BGR2RGB(),
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
        ]),
    }

for subj_left_id, subj_left_out in enumerate(folds):
    np.random.seed(7)
    test_subj = subj_left_out
    train_id= [*range(len(folds))]
    train_id.pop(subj_left_id)
    val_id = random.choice(train_id)
    print(val_id)
    val_subj = folds[val_id]

    if os.path.isfile('./results_sf' + str(7) + '/' + str(subj_left_id) + '.npz'):
        print('continue')
        #continue
    # Initialize the model for this run
    
    model_ft, input_size = initialize_model('face1_vgg', 10, feature_extract=False, use_pretrained=True)

    print('-'*10 + "cross-validation: " + 
    "(" + str(subj_left_id+1) + "/5)" + '-'*10)
    datasets = {x: McMasterDataset(image_dir, label_dir, val_subj, test_subj, x, data_transforms[x]) for x in ['train', 'val', 'test']}
    
    print(datasets['val'].__len__)
    print(datasets['train'].__len__)
    print(datasets['test'].__len__)

    weights = {}
    for phase in ['train', 'val', 'test']:
        labels = [x['framePSPI'] for x in datasets[phase]]
        labels = np.stack(labels)
        classes, classweights = np.unique(labels, return_counts=True)
        classweights = np.reciprocal(classweights.astype(float))
        sampleweights = classweights[np.searchsorted(classes, labels)]
        classweights = classweights * sampleweights.shape[0] / np.sum(sampleweights) 
        weights[phase] = {'classes':classes, 'classweights': classweights}
