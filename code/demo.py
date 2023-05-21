import sys
sys.path.append('./code')
import argparse
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import argparse
from vgg_face import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
plt.switch_backend('agg')

# to download face crop network, please go to:
# https://github.com/opencv/opencv/tree/master/data/haarcascades 
# to download vgg_face model .pth file, please go to:
# https://drive.google.com/drive/folders/1wzqoBauX746f9YxpFrmf8TUlUhfb8vDN?usp=share_link
# model path:
# /Volumes/Yuan-T7/FAU_models/models_r10/checkpoint_epoch69.pth
# /Volumes/Yuan-T7/FAU_models/checkpoint_epoch_init.pth

parser = argparse.ArgumentParser(description='DEMO')
parser.add_argument('--mpath', '-p', default='/Volumes/Yuan-T7/FAU_models/models_r11/checkpoint_epoch69_11.pth', type=str, 
                    help='the path of model')
parser.add_argument('--scale', '-s', default=80, type=int, 
                    help='determine the face crop size')
parser.add_argument('--record', '-r', default='off', choices=['on', 'off'], type=str, 
                    help='determine if recording is on or off')
parser.add_argument('--record_dir', '-d', default='record_0', type=str, 
                    help='define the directory name for video save')
parser.add_argument('--round', default=1, type=int, choices=[1, 0], 
                    help='define if to show facial detection indicator')
parser.add_argument('--square', default=0, type=int, choices=[1, 0], 
                    help='define if to show face cropping indicator')
parser.add_argument('--sampling_rate', default=1, choices=[1, 2, 3, 4, 5], type=int, 
                    help='determine the speed of samples fed into network. larger the value, lower the speed. if using a low-end device, use large value')
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
        output = output * torch.FloatTensor([16] + [5]*9 + [25]).to(device)
    return output

def plot_bar(au_scores, title):
    #labels = ['AU4', 'AU6', 'AU7', 'AU9', 'AU10', 'AU12', 'AU20', 'AU25', 'AU26', 'AU43']
    labels = ['Brow\nLowerer', 'Cheek\nRaiser', 'Lid\nTightener', 'Nose\nWrinkler', 'Upper Lip\nRaiser', 'Lip Corner\nPuller', 'Lip\nStretcher', 'Lips\nPart', 'Jaw\nDrop', 'Eyes\nClosed']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'burlywood']
    fig, ax = plt.subplots(figsize=(10, 10.8))
    values = au_scores[1:]
    fig, ax = plt.subplots(figsize=(10, 10.8))

    #ax.barh(labels, values, color=colors)
    ordertoshow=[7, 6, 5, 4, 3, 2, 8, 0, 1, 9]
    ax.barh(np.take(labels, ordertoshow), np.take(values, ordertoshow), color=np.take(colors, ordertoshow))

    ax.tick_params(axis='both', labelsize=15)
    ax.set_title(title + ' PSPI {:.2f}/16'.format(au_scores[0]), fontsize=20)
    ax.set_xlim([0, 5])
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    plot = np.array(canvas.renderer.buffer_rgba())
    plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
    plt.close()
    return plot

def rounded_rectangle(image, x, y, w, h, radius, thickness):
    cv2.ellipse(image, (x + radius, y + radius), (radius, radius), 180, 0, 90, (51,255,255), thickness)
    cv2.ellipse(image, (x + w - radius, y + radius), (radius, radius), 270, 0, 90, (51,255,255), thickness)
    cv2.ellipse(image, (x + radius, y + h - radius), (radius, radius), 90, 0, 90, (51,255,255), thickness)
    cv2.ellipse(image, (x + w - radius, y + h - radius), (radius, radius), 0, 0, 90, (51,255,255), thickness)
    return image

def adaptive_canvas(image_left, image_right):
    h_l, w_l, _ = image_left.shape
    h_r, w_r, _ = image_right.shape
    canvas_h = max(h_l, h_r)
    canvas_w = w_l+w_r
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h_l, :w_l, :] = image_left[:,:,:3]
    canvas[:h_r, w_l:, :] = image_right[:,:,:3]
    return canvas

def main():
    # add DL model
    num_classes = 11
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
    cv2.resizeWindow('face_cap', 1500, 600)

    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap_info = '({}*{}, fps {})'.format(cap_w, cap_h, fps)

    # create directory for 
    if args.record == 'on':
        save_path = './{}/'.format(args.record_dir)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    save_count = 0
    while(True):
        ret, frame = cap.read()
    
        # mirror the video
        frame_flip = cv2.flip(frame,1)
        
        # opencv2 detect face
        frame_gray = cv2.cvtColor(frame_flip, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(frame_gray, 1.1, 6) 
        for (x, y, w, h) in faces:
            h = w
            crop_frame = np.copy(frame_flip)[y-2*args.scale:y+h+args.scale, x-args.scale:x+w+args.scale]
            if args.round == 1:
                frame_flip = rounded_rectangle(frame_flip, x, y, w, h, 100, 20)
            if args.square == 1: 
                frame_flip = cv2.rectangle(frame_flip, (x-args.scale, y-2*args.scale), (x+w+args.scale, y+h+args.scale), (255, 255, 255), 4)
        
        # feed the frame into the model
        try: 
            if save_count % args.sampling_rate == 0:
                output = test_model(model, crop_frame, device)
                output = output.flatten()
                output_cpu = torch.Tensor.cpu(output).numpy()
                output_plot = plot_bar(output_cpu, cap_info)
                output_text = '|PSPI {:.2f} |au4 {:.2f} |au6 {:.2f} |au7 {:.2f} |au9 {:.2f} |au10 {:.2f} |au12 {:.2f} |au20 {:.2f} |au25 {:.2f} |au26 {:.2f} |au43 {:.2f}|'.format(output[0].item(), output[1].item(), output[2].item(), output[3].item(), output[4].item(), output[5].item(), output[6].item(), output[7].item(), output[8].item(), output[9].item(), output[10].item())

            # save video option
            if args.record == 'on':
                filename = save_path + str(save_count) + '.png'
                cv2.imwrite(filename, crop_frame)
                np.save(save_path + str(save_count) + '.npy', output_cpu)
                save_count += 1
            else:
                pass

        except Exception as error:
            output_plot = canvas = np.ones((1080, 1000, 3), dtype=np.uint8)*255
            output_text = str(error)
        
        # print AU info at the bottom of each frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        #thickness = 2
        text_size, _ = cv2.getTextSize(output_text, font, font_scale, 2)
        text_x = int((cap_w - text_size[0]) / 2)
        text_y = cap_h - text_size[1] - 10
        cv2.putText(frame_flip, output_text, (text_x, text_y), font, font_scale, (255, 255, 255), 2)

        # dispay canvas
        canvas = adaptive_canvas(frame_flip, output_plot)
        plt.close()
        cv2.imshow('face_cap', canvas)

        # Press 'Q' to quit camera
        if cv2.waitKey(1) == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    main()