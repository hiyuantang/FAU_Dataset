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

def main():
    # define a video capture object
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cv2.namedWindow('MyWindow', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('MyWindow', 600, 400)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        
        # mirror the video
        frame_flip = cv2.flip(frame,1)
        
        # opencv2 detect face
        frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 4) 
        for (x, y, w, h) in faces:
            cv2.rectangle(frame_flip, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('MyWindow', frame_flip)
        # Press 'Q' to quit camera
        if cv2.waitKey(1) == ord('q'):
            break
    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()