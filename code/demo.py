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
import random
import argparse
from vgg_face import *
import matplotlib.pyplot as plt
from facegenDataset import *
from FAUDataset import *
plt.switch_backend('agg')

parser = argparse.ArgumentParser(description='DEMO')
parser.add_argument('--mpath', '-p', default='/Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth', type=str, 
                    help='transfer training by defining the path of stored weights')
args = parser.parse_args()

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

def test_model(model, crop_frame, device, criterion, batch_size):
    model.eval()
    with torch.no_grad():
        input = crop_frame.to(device)
        output = model(input)
        output = output * torch.FloatTensor([16] + [5]*9).to(device)
    return output

def main():
    # add DL model
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = VGG_16().to(device)
    set_parameter_requires_grad(model, feature_extracting=False)
    num_ftrs = model.fc8.in_features
    model.fc8 = nn.Linear(num_ftrs, num_classes).to(device)
    model.load_state_dict(torch.load(args.mpath, map_location=device))
    
    data_transforms = {
            'test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                #transforms.Normalize([0.3873985 , 0.42637664, 0.53720075], [0.2046528 , 0.19909547, 0.19015081])
            ])
        }

    # Setup the loss fxn
    criterion = nn.MSELoss(reduction='none')


    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.namedWindow('face_cap', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('face_cap', 900, 600)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        
        # mirror the video
        frame_flip = cv2.flip(frame,1)
        
        # opencv2 detect face
        frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 6) 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_flip, (x, y), (x+w, y+h), (255, 0, 0), 2)
            crop_frame = frame_flip[y:y+h, x:x+w]
        try: 
            cv2.imshow('face_cap', crop_frame)
        except:
            cv2.imshow('face_cap', frame_flip)
        
        # feed the frame into the model
        output = test_model(model, crop_frame, device, criterion, batch_size=1)

        # Press 'Q' to quit camera
        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()