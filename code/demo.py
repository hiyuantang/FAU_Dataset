import sys
sys.path.append('./code')
import argparse
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from vgg_face import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# to download face crop network, please go to:
# https://github.com/opencv/opencv/tree/master/data/haarcascades 
# to download vgg_face model .pth file, please go to:
# https://drive.google.com/drive/folders/1wzqoBauX746f9YxpFrmf8TUlUhfb8vDN?usp=share_link
# model path:
# /Volumes/Yuan-T7/FAU_models/models_r10/checkpoint_epoch69.pth
# /Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth

parser = argparse.ArgumentParser(description='DEMO')
parser.add_argument('--mpath', '-p', default='/Volumes/Yuan-T7/FAU_models/models_r10/checkpoint_epoch69.pth', type=str, 
                    help='transfer training by defining the path of stored weights')
parser.add_argument('--scale', '-s', default=80, type=int, 
                    help='determine the face crop size')
args = parser.parse_args()

def set_parameter_requires_grad(model, feature_extracting):
    for param in model.parameters():
        param.requires_grad = False if feature_extracting else True

def centercrop_opencv(image, size):
    height, width, _ = image.shape
    crop_height = crop_width = size
    start_height = (height - crop_height) // 2
    start_width = (width - crop_width) // 2 
    cropped_image = image[start_height:start_height+crop_height, start_width:start_width+crop_width]
    return cropped_image

def test_model(model, crop_frame, device):
    model.eval()
    with torch.no_grad():
        crop_frame = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)
        crop_frame = cv2.resize(crop_frame, (256, 256))
        crop_frame = centercrop_opencv(crop_frame, 224)
        crop_frame = crop_frame / 255.0
        crop_frame = np.transpose(crop_frame, (2, 0, 1))
        crop_frame = np.expand_dims(crop_frame, axis=0)  # add batch dimension
        crop_frame = torch.from_numpy(crop_frame).float().to(device)
        output = model(crop_frame)
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

    # add face detection model, choose between:
    # haarcascade_frontalface_default.xml
    # haarcascade_frontalface_alt_tree.xml is preferred
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt_tree.xml')

    # set up a camera window
    cv2.namedWindow('face_cap', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('face_cap', 1200, 900)
    cap = cv2.VideoCapture(0)
    while(True):
        ret, frame = cap.read()
        
        # mirror the video
        frame_flip = cv2.flip(frame,1)
        
        # opencv2 detect face
        frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 6) 
        for (x, y, w, h) in faces:
            h = w
            cv2.rectangle(frame_flip, (x-args.scale, y-2*args.scale), (x+w+args.scale, y+h+args.scale), (255, 0, 0), 4)
            crop_frame = frame_flip[y-args.scale:y+h+args.scale*2, x-args.scale:x+w+args.scale*2]
        
        # feed the frame into the model
        try: 
            output = test_model(model, crop_frame, device)
            output = output.flatten()
            output_text = '|PSPI {:.2f} |au4 {:.2f} |au6 {:.2f} |au7 {:.2f} |au10 {:.2f} |au12 {:.2f} |au20 {:.2f} |au25 {:.2f} |au26 {:.2f} |au43 {:.2f}|'.format(output[0].item(), output[1].item(), output[2].item(), output[3].item(), output[4].item(), output[5].item(), output[6].item(), output[7].item(), output[8].item(), output[9].item())
        except:
            output_text = 'No Face Is Detected'
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        #thickness = 2
        text_size, _ = cv2.getTextSize(output_text, font, font_scale, 2)
        height, width, channels = frame_flip.shape
        text_x = int((width - text_size[0]) / 2)
        text_y = height - text_size[1] - 10
        cv2.putText(frame_flip, output_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

        cv2.imshow('face_cap', frame_flip)

        # Press 'Q' to quit camera
        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()