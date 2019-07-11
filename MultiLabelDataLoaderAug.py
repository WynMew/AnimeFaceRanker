from __future__ import print_function
import os
import sys
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np

def gauss_noise(image,var):
    image=np.array(image)
    row,col,ch= image.shape
    mean = 0
    sigma = var
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noise = image + gauss.astype(int)
    absnoise = abs(noise)
    absnoise[absnoise > 256] = 255
    noise=Image.fromarray(absnoise.astype('uint8'))
    return noise

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    image = np.array(image)
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    output = Image.fromarray(output.astype('uint8'))
    return output



def random_flip(img):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def random_noise(img):
    tmp = random.random()
    if tmp < 0.05:
        var = random.random()*10
        img =  gauss_noise(img,var)
    elif tmp>0.95:
        var = random.random()*0.03
        img = sp_noise(img,var)
    else:
        pass
    return img

def random_noise_Para(img, pGauss, vGauss, pSP, vSP):
    tmp = random.random()
    '''
    if tmp < 0.05:
        var = random.random()*10
        img =  gauss_noise(img,var)
    elif tmp>0.95:
        var = random.random()*0.03
        img = sp_noise(img,var)
    else:
        pass
    '''
    if tmp < pGauss:
        var = random.random()*vGauss
        img =  gauss_noise(img,var)
    elif tmp>(1-pSP):
        var = random.random()*vSP
        img = sp_noise(img,var)
    else:
        pass
    return img


def resize(img, boxes, size, max_size=1000, random_interpolation=False):
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h

    method = random.choice([
        Image.BOX,
        Image.NEAREST,
        Image.HAMMING,
        Image.BICUBIC,
        Image.LANCZOS,
        Image.BILINEAR]) if random_interpolation else Image.BILINEAR
    img = img.resize((ow,oh), method)
    if boxes is not None:
        boxes = boxes * torch.Tensor([sw,sh,sw,sh])
    return img, boxes

class ListDatasetLite(data.Dataset):
    def __init__(self, root, list_file, classes=None, barcode_Dict = None, transform=None):
        self.root = root
        self.num_classes = len(classes)
        self.barcodeDict = barcode_Dict
        self.transform = transform
        self.fnames = []
        self.labels = []
        self.class_to_ind = dict(zip(classes, range(len(classes))))
        self.fill=(0,0,0)

        if isinstance(list_file, list):
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            label = []
            c = splited[5]
            code_c = self.barcodeDict[c]
            # print('code_c', code_c)
            label.append(self.class_to_ind[code_c])
            self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):

        fname = self.fnames[idx].strip()
        #print(fname)
        #print(os.path.join(self.root, fname))
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        labels = self.labels[idx].clone()

        if self.transform:
            img, labels = self.transform(img, labels)
        target = []

        for iTarget in range(0, self.num_classes):
            PositiveLabel = False
            for item in labels:
                if (iTarget == item):
                    PositiveLabel = True
            if (PositiveLabel):
                target.append([1])
            else:
                target.append([0])
        target = torch.FloatTensor(target)
        #print(fname, target)
        return img, target

    def __len__(self):
        return self.num_imgs
