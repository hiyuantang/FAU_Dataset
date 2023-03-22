# -*- coding: utf-8 -*-

from __future__ import print_function

''' directory: /mnt/cube/projects/yundong/pain_detection  '''
import cv2
import os
import numpy as np
import scipy.io as sio
import pickle
from os import listdir
from os.path import isfile, join, isdir
import sys
'''
This file contains helper functions that are required from McMasterDataset.py
'''
def printProgressBar_py3 (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def printProgressBar_py2(iteration, total, prefix='', suffix='', decimals=1, length=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def get_video_to_image_id(image_dir):
    '''
    Dump a dictionary that get's all the image names given the video name
    key - value : gf097t1aaaff - [gf097t1aaaff001.png, gf097t1aaaff002.png, ... , gf097t1aaaff518.png]
    '''

    # if it exist, do nothing
    if os.path.isfile('/mnt/cube/projects/yundong/pain_detection/video_to_image_id'):
        return

    # Create video to image dict
    video_to_image_id = {}
    for subject in listdir(image_dir):
        if isdir(join(image_dir, subject)):
            for video in listdir(join(image_dir, subject)):
                if isdir(join(image_dir, subject, video)):
                    image_id_list = []
                    for image in listdir(join(image_dir, subject, video)):
                        if image[-3:] == 'png':
                            image_id_list.append(image)
                    video_to_image_id[video] = sorted(image_id_list)

    # sanity check
    assert sum([len(video_to_image_id[key]) for key in video_to_image_id.keys()]) == 47397
    if sys.version_info[0] < 3:
        pickle.dump(video_to_image_id, open('/mnt/cube/projects/yundong/pain_detection/video_to_image_id','wb'), protocol=2)
    else:
        pickle.dump(video_to_image_id, open('/mnt/cube/projects/yundong/pain_detection/video_to_image_id','wb'))

def binarySearch( arr, l, r, x):
    '''
    Helper function for searching index of the element (as a string)
    '''
    # Check base case
    if r >= l:
        mid = int(l + (r - l)/2)

        if arr[mid] == x:
            return mid

        elif arr[mid] > x:
            return binarySearch( arr, l, mid-1, x)

        else:
            return binarySearch( arr, mid + 1, r, x)
    else:
        # Element is not present in the array
        raise NameError('Element is not presented in the array.')

def read_AAM(aam_type, file_path):
    '''
    read AAM file depending on the file type (txt or mat)
    '''
    aam = []
    if aam_type == 'txt':
        name = os.path.join(file_path + '_aam')
        f = open(name + '.txt', "r")
        scorestr = f.readlines()
        f.close()

        for line in scorestr:
            words = [x.strip() for x in line.split(' ') if x]
            aam = aam + [float(words[0]), float(words[1])]
        aam = np.stack(aam)
    else:
        name = os.path.join(file_path + '_aam.mat')
        aam = sio.loadmat(name)['aam_proc']
        aam = (aam - np.array([[169.67,144.99]])) / np.array([[31.81,34.25]])
    return aam.flatten()

def read_feature(path):
    '''
    read feature map
    '''
    feature = np.load(path + '.npz')["feat_map"]
    return feature

def image_id_to_image_seq(image_id, left_window_len, right_window_len):
    '''
    Given the window length (left and right) of the sequence segment, create
    a dictionary that the key is the image ID and the value is the image ID's
    around it.

    returns: (image_segment) - A list containing the image names surrounding the center images
            (edge_img) - the name of the image that is on the edge
    '''

    null_id = "_*_"
    video_to_image_id = pickle.load(open('/mnt/cube/projects/yundong/pain_detection/video_to_image_id', 'rb'))
    image_seq = video_to_image_id[image_id[:-7]]
    image_index = binarySearch(image_seq, 0, len(image_seq) - 1, image_id)
    image_segment = []
    edge_img = None
    if image_index < left_window_len:
        # pad left
        for i in range(left_window_len - image_index):
            image_segment.append(null_id)
        edge_img = image_seq[0]
    # add left image id
    for i in range(max(0,image_index - left_window_len), image_index):
        image_segment.append(image_seq[i])
    # add itself
    image_segment.append(image_id)
    # add right image id
    right_idx = min(len(image_seq), image_index + right_window_len + 1)
    for i in range(image_index + 1, right_idx):
        image_segment.append(image_seq[i])

    # pad right
    for i in range(image_index + right_window_len + 1 - len(image_seq)):
        image_segment.append(null_id)
        edge_img = image_seq[-1]
    return image_segment, edge_img

def read_labels(path):
    '''
    Read only one type of sequence label given path.
    '''

    f=open(path + '.txt', "r")
    video_label = float(f.readline().strip())
    f.close()
    return np.array([video_label])

def load_process_image(path, transform):
    '''
    Read and transform images
    '''

    image = cv2.imread(path)
    if transform:
        return transform(image)
    else:
        return image

def read_au(path, dtype = 'txt'):
    '''
    Read AU
    '''
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

def read_rnn_seg(args):
    '''
    Read segmented seq for RNN.
    '''

    rnn_kwargs, imagepath, image_id, subj_id, video_id, AAM_filetype, AAMpath, feat_path, AUpath, PSPIpath = args
    rnn_input_segment = None
    rnn_output_segment = None
    input_path = None
    if rnn_kwargs != None:
        try:
            left_window_len = rnn_kwargs['left_window_len']
            right_window_len = rnn_kwargs['right_window_len']
            input_type = rnn_kwargs['input_type']
            padding_type = rnn_kwargs['padding_type']
            use_au = rnn_kwargs['use_au']
        except KeyError:
            print('Invalid RNN arguments.')
            exit()

        rnn_output_segment = np.empty((0, 10)) if use_au else np.empty((0, 1))
        get_video_to_image_id(imagepath) # create _get_video_to_image_id dictionary
        image_seq_segment, edge_img = image_id_to_image_seq(image_id, left_window_len, right_window_len)

        # When the segment type is AAM

        if input_type == 'AAM':
            rnn_input_segment = np.empty((0,132))
            input_path = AAMpath
        elif input_type == 'feat_4096':
            rnn_input_segment = np.empty((0,4096))
            input_path = feat_path
        for image_name in image_seq_segment:
            # dealing with edge cases
            if image_name == '_*_':

                #padding strategy 1
                if padding_type == 'zero_padding':
                    if input_type == 'AAM':
                        rnn_input_segment = np.vstack((rnn_input_segment, np.zeros((1,132))))
                    elif input_type == 'feat_4096':
                        rnn_input_segment = np.vstack((rnn_input_segment, np.zeros((1,4096))))

                    rnn_output_segment = np.vstack((rnn_output_segment, np.zeros((1,10)))) \
                        if use_au else np.vstack((rnn_output_segment, np.zeros((1,1))))

                # padding strategy 2
                elif padding_type == 'repeat_edge':
                    #### read input ####
                    if input_type == 'AAM':
                        edge_aam = read_AAM(AAM_filetype, os.path.join(input_path,subj_id, video_id, edge_img[:-4]))
                        rnn_input_segment = np.vstack((rnn_input_segment, edge_aam.reshape(1,-1)))
                    elif input_type == 'feat_4096':
                        edge_feat = read_feature(os.path.join(input_path,subj_id, video_id, edge_img[:-4]))
                        rnn_input_segment = np.vstack((rnn_input_segment, edge_feat.reshape(1,-1)))

                    #### read labels #####

                    pspi = read_labels(os.path.join(PSPIpath,subj_id,video_id,edge_img[:-4]+'_facs'))
                    # whether the labels contain au
                    if use_au:
                        au = read_au(os.path.join(AUpath,subj_id,video_id,edge_img[:-4]+'_facs'))
                        pspi_au = np.concatenate((au, pspi))
                        rnn_output_segment = np.vstack((rnn_output_segment, pspi_au.reshape(1,-1)))
                    else:
                        rnn_output_segment = np.vstack((rnn_output_segment, pspi.reshape(1,-1)))

                # strategy 3: ignore this sample
                elif padding_type == 'discard':
                    return 'discard', 'discard'
                else:
                    print('Error. Invalid padding type')
                    exit()
            else:

                ### read input ###
                if input_type == 'AAM':
                    seg_aam = read_AAM(AAM_filetype, os.path.join(input_path,subj_id, video_id, image_name[:-4]))
                    rnn_input_segment = np.vstack((rnn_input_segment, seg_aam.reshape(1,-1)))
                elif input_type == 'feat_4096':
                    seg_feat = read_feature(os.path.join(input_path,subj_id, video_id, image_name[:-4]))
                    rnn_input_segment = np.vstack((rnn_input_segment, seg_feat.reshape(1,-1)))

                ### read labels ###
                seg_pspi = read_labels(os.path.join(PSPIpath,subj_id,video_id,image_name[:-4]+'_facs'))
                if use_au:
                    seg_au = read_au(os.path.join(AUpath,subj_id,video_id,image_name[:-4]+'_facs'))
                    seg_pspi_au = np.concatenate((seg_au, seg_pspi))
                    rnn_output_segment = np.vstack((rnn_output_segment, seg_pspi_au.reshape(1,-1)))
                else:
                    rnn_output_segment = np.vstack((rnn_output_segment, seg_pspi.reshape(1,-1)))


    return rnn_input_segment, rnn_output_segment

def my_collate(batch):
    '''
    filtering out discarded sample in a batch
    '''
    import torch.utils.data.dataloader as dataloader
    new_batch = list(filter(lambda x:type(x['rnn_seq'][0]) is not str, batch))
    if len(new_batch) == 0:
        return None

    return dataloader.default_collate(new_batch)

''' For MDV'''
def get_PSPI(aus):
    ''' get PSPI from aus
        return a 1-d array
    '''
    aus = np.reshape(aus, (-1,9))
    return aus[:,0] + np.max(aus[:,1:3], axis=1) + aus[:,3] + aus[:,8]


def get_stat_feature(framePSPI):
    ''' get mean, max, min, std, 95 percentile, 85, 75, 50, 25, half_rect_mean of list
        axis 0 = number of AUs, axis 1 = number of features
        input: framePSPI - (N, D), where N is the number of image labels, D is the number of AUs
        return: feature - (D, 9)
    '''
    framePSPI = np.asarray(framePSPI)
    feature = [np.nanmean(framePSPI, axis=0),
        np.nanmax(framePSPI, axis=0),
        np.nanmin(framePSPI, axis=0),
        np.nanstd(framePSPI, axis=0),
        np.nanpercentile(framePSPI, 95, axis=0),
        np.nanpercentile(framePSPI, 85, axis=0),
        np.nanpercentile(framePSPI, 75, axis=0),
        np.nanpercentile(framePSPI, 50, axis=0),
        np.nanpercentile(framePSPI, 25, axis=0)]
    feature = np.swapaxes(np.asarray(feature), 0, 1)
    return feature
