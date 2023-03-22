from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from McMasterDataset_func import *
import torch
import os
import cv2
import time
import numpy as np
import sys
from scipy import stats

class McMasterDataset(Dataset):
    """ McMaster Shoulder Pain Dataset. """

    def __init__(self, image_dir, label_dir, val_subj_id, test_subj_id, subset, transform=None, preloading=True, pred_dir=None):
        """
        Args:
            image_dir (string): Path to the image data "UNBCMcMaster_cropped/Images0.3"
            label_dir (string): Path to the label (pain level, etc.) "UNBCMcMaster"
            val_subj_id ([string]): list of paths containing validation data
            test_subj_id ([string]): list of paths containing test data
            subset (string): train, val, test
            transform (callable, optional): Optional transfomr to be applied on a sample
            preloading (optional): True if want to load data at initialization
            pred_dir (optional): Path to the prediction of the frame "newnorm_PSPIAU"
        """
        self.seqVASpath = os.path.join(label_dir, 'Sequence_Labels','VAS')
        self.frameVASpath = os.path.join(label_dir, 'Frame_Labels','PSPI')
        self.AUpath = os.path.join(label_dir, 'Frame_Labels', 'FACS')
        self.AAMpath = os.path.join(label_dir, 'AAM_landmarks')
        self.imagepath = image_dir
        self.subset = subset
        self.image_files = [(root, name) for root,dirs,files in os.walk(self.imagepath) for name in sorted(files) if name[-3:]=='png' and 
         ((name[2:5] in test_subj_id and subset=='test') or
          (name[2:5] in val_subj_id and subset=='val') or 
          (not(name[2:5] in val_subj_id+test_subj_id) and subset=='train'))]
        self.transform = transform
        self.preloading = preloading
        self.framescorepath = pred_dir
        self.huge_dictionary = self.preload() if preloading else None

    def __len__(self):
        return sum([len(self.image_files)])

    def preload(self):
        print('Preloading %s data...'%self.subset)
        tic = time.time()
        huge_dictionary = []

        if sys.version_info[0] >= 3:
            printProgressBar_py3(0, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)
        else:
            printProgressBar_py2(0, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)

        for idx in range(self.__len__()):
            sample = self.get_item_helper(idx)
            huge_dictionary.append(sample)

            if sys.version_info[0] >= 3:
                printProgressBar_py3(idx + 1, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)
            else:
                printProgressBar_py2(idx + 1, self.__len__(), prefix = 'Progress:', suffix = 'Complete', length = 50)

        toc = time.time()
        print('Finished in %.2f minutes.' %float((toc - tic)/60))
        return huge_dictionary
        # for key, value in huge_dictionary.items():
        #     pickle.dump(huge_dictionary, open(key+'_'+self.subset, 'wb'))

    def get_item_helper(self, idx):
        """
        Return: sample
            an example of sample:
                sample['image'] = np.ndarray WxHx3
                sample['image_id'] = 'gf097t1aaaff001.png'
                sample['video_id'] = 'gf097t1aaaff'
                sample['au'] = np.ndarray (9,)
                sample['framelabel'] = 0
                sample['subj_id'] = '097-gf097'
                sample['videoVAS] = 5.0
                sample['framePSPI'] = 0.0
                sample['aam'] = np.ndarray (132,)
        """
        img_dir = self.image_files[idx][0]
        img_name = self.image_files[idx][1]
        image = cv2.imread(os.path.join(img_dir, img_name))
        image_id = img_name
        video_dir = os.path.split(img_dir)
        video_id = video_dir[1]
        subj_dir = os.path.split(video_dir[0])
        subj_id = subj_dir[1]

        name = os.path.join(self.seqVASpath,subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        # scorestr = scorestr.lstrip()
        videoVAS = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))

        name = os.path.join(os.path.split(self.seqVASpath)[0], 'SEN',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoSEN = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'OPR',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoOPR = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        
        name = os.path.join(os.path.split(self.seqVASpath)[0], 'AFF',subj_id,video_id)
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        videoAFF = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))
        # framePSPI
        name = os.path.join(self.frameVASpath,subj_id,video_id,img_name[:-4]+'_facs')
        f=open(name + '.txt', "r")
        scorestr = f.read()
        f.close()
        framePSPI = float(scorestr[0:scorestr.find('e')]) * (10** int(scorestr[scorestr.find('+')+1:]))   
        framelabel = 0+(framePSPI > 0)

        # frameAU
        name = os.path.join(self.AUpath,subj_id,video_id,img_name[:-4]+'_facs')
        f=open(name + '.txt', "r")
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

        # frame prediction
        if self.framescorepath: # prediction framePSPI instead
            name = os.path.join(self.framescorepath,subj_id,video_id,img_name[:-4])
            if os.path.isfile(name + '.npz'):
                pred = np.load(name + '.npz')['output']
            elif os.path.isfile(name + '.npy'):
                pred = np.load(name + '.npy')
        else:
            pred = 0


        au10 = au[[3,5,6,8,9,11,19,24,25,42]]
        au = au[[3,5,6,9,11,19,24,25,42]]
        # au = au[[3,5,6,8,9,11,19,24,25,26,42]]

        # frameAAM
        name = os.path.join(self.AAMpath,subj_id, video_id,img_name[:-4]+'_aam')
        f = open(name + '.txt', "r")
        scorestr = f.readlines()
        f.close()
        aam = []
        for line in scorestr:
            words = [x.strip() for x in line.split(' ') if x]
            aam = aam + [float(words[0]), float(words[1])]
        aam = np.stack(aam)
        # aam = None

        # sample = {'image': image, 'au': au, 'framePSPI': framePSPI}
        sample = {'image': image, 'image_id': image_id, 'video_id': video_id, 'au': au, 'au10': au10,'aam': aam,
            'framelabel': framelabel, 'subj_id': subj_id, 'videoVAS': videoVAS, 
            'videoAFF': videoAFF, 'videoOPR': videoOPR, 
            'videoSEN': videoSEN,'framePSPI': framePSPI, 'pred': pred}
        if self.transform:
            sample['image'] = self.transform(image)

        return sample

    def __getitem__(self, idx):
        sample = {}
        if self.preloading:
            sample = self.huge_dictionary[idx]
        else:
            sample = self.get_item_helper(idx)
        return sample

class BGR2RGB(object):
    """Convert BGR image to RGB.
    """

    def __call__(self, image):
        image = image[:,:,::-1]

        return image

class prewhiten(object):
    def __call__(self, x):
        mean = x.mean()
        std = x.std()
        std_adj = std.clamp(min=1.0/(float(x.numel())**0.5))
        y = (x - mean) / std_adj
        return y

class darken(object):
    """Darken image by parameter p
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        image = (image * self.p).astype(np.uint8)
        return image

class zscore_matching(object):
    """ Histogram Matching to target_hist
    """

    def __init__(self, mean, std, excludewhite = True):
        self.mean = mean
        self.std = std
        self.excludewhite = excludewhite

    def __call__(self, image):
        # image = match_histogram(image, self.target_hist)
        hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        zscore_img = hsv_img[:,:,2].astype(np.float)
        if self.excludewhite:
            idx = zscore_img<255
        else:
            idx = np.ones(zscore_img.shape)
        zscore_img[idx] = stats.zscore(zscore_img[idx])
        matched_img_v = zscore_img
        matched_img_v[idx] = (zscore_img[idx] * self.std + self.mean)
        matched_img_v[matched_img_v>255] = 255
        matched_img_v[matched_img_v<0] = 0    
        hsv_img[:,:,2] = matched_img_v
        matched_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
        return matched_img