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

def pasteandrotate(img, neg, negroot, margin=100):
    fill = (0, 0, 0)
    w, h = img.size
    if w>h:
        ow, oh = w, w
        square = Image.new('RGB', (ow, oh), fill)
        square.paste(img, (0, int((w-h)/2)))
    else:
        ow, oh = h, h
        square = Image.new('RGB', (ow, oh), fill)
        square.paste(img, (int((h-w)/2), 0))
    tmp = random.random()
    if tmp < 0.1:
        img = square.rotate(90)
    elif tmp> 0.9:
        img = square.rotate(-90)
    else:
        img= square
    tmp = random.random()
    if tmp < 0.4:
        w, h = img.size
        ow, oh = w+margin*2, h + margin*2
        canvas = Image.new('RGB', (ow, oh), fill)
        canvas.paste(img, (margin, margin))
        return canvas
    elif tmp>0.8:
        negMargin = int(random.uniform(150, 400)) # random background margin
        line = neg[int(random.uniform(1, len(neg)))]
        NegFile = line.split(' ')[0]
        w, h = img.size
        ow, oh = w + negMargin, h + negMargin
        canvas = Image.new('RGB', (ow, oh), fill)
        NegImg = Image.open(os.path.join(negroot, NegFile))
        if NegImg.mode != 'RGB':
            NegImg = NegImg.convert('RGB')
        negw, negh = NegImg.size
        xmin = random.uniform(0, max(10, negw - 50))
        ymin = random.uniform(0, max(10, negh - 50))
        xmax = 0
        ymax = 0
        flag = True
        while flag:
            xmax = random.uniform(xmin, negw)
            ymax = random.uniform(ymin, negh)
            if xmax - xmin > 50 and ymax - ymin > 50:
                flag = False
        NegImg_c = NegImg.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        NegImg = NegImg_c.resize((ow,oh))
        canvas.paste(NegImg, (0, 0))
        x_off = int(random.uniform(0, negMargin))
        y_off = int(random.uniform(0, negMargin))
        canvas.paste(img, (x_off, y_off))
        return canvas
    else:
        w, h = img.size
        ow, oh = w+margin*2, h + margin*2
        canvas = Image.new('RGB', (ow, oh), fill)
        x_off = int(random.uniform(0,margin*2))
        y_off = int(random.uniform(0,margin*2))
        canvas.paste(img, (x_off, y_off))
        return canvas


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

def random_paste_mix(img, labels, neg, crop, classes = None, barcode_Dict = None,
                     negroot='/home/wynmew/workspace/Data',croproot='/home/wynmew/workspace/Data/17class/', margin=100):
    # img : <class 'PIL.JpegImagePlugin.JpegImageFile'>
    #im1 = Image.open('/Users/rem7/Desktop/_1.jpg')
    #im2 = Image.open('/Users/rem7/Desktop/_2.jpg')
    #Image.blend(im1, im2, 0.5).save('a.jpg')
    if random.random() < 0.3 and labels.cpu().numpy()[0] != 0:
        class_to_ind = dict(zip(classes, range(len(classes))))
        cropfile = crop[int(random.uniform(1, len(crop)))]
        mixImg = Image.open(os.path.join(croproot, cropfile))
        w, h = mixImg.size
        if w > h:
            ow, oh = w, w
            square = Image.new('RGB', (ow, oh), (0,0,0))
            square.paste(mixImg, (0, int((w - h) / 2)))
        else:
            ow, oh = h, h
            square = Image.new('RGB', (ow, oh), (0,0,0))
            square.paste(mixImg, (int((h - w) / 2), 0))
        Ang = int(random.uniform(0, 359))
        mixImg = square.rotate(Ang)
        mx = max(img.size[0],mixImg.size[0])
        my = max(img.size[1],mixImg.size[1])
        c1 = Image.new('RGB', (mx, my), (0,0,0))
        c1.paste(img,(int((mx-img.size[0])/2),int((my-img.size[1])/2)))
        c2 = Image.new('RGB', (mx, my), (0, 0, 0))
        c2.paste(mixImg, (int((mx - mixImg.size[0]) / 2), int((my - mixImg.size[1]) / 2)))
        mixRatio = random.uniform(0.4,0.6)
        img=Image.blend(c1, c2, mixRatio)
        splited = cropfile.strip().split('/')
        c = splited[1]
        label=class_to_ind[barcode_Dict[c]]
        cl = []
        cl.append(label)
        if labels.cpu().numpy()[0] != label:
            labels = torch.cat((labels, torch.LongTensor(cl)), 0)
        #    print('labels2 = ', labels)
        #else:
        #   print('labels2_el = ', labels)

    fill = (0, 0, 0)
    tmp=random.random()
    #print('labels = ', labels) #torch.LongTensor of size 1
    if tmp < 0.4:
        w, h = img.size
        ow, oh = w+200, h + 200
        canvas = Image.new('RGB', (ow, oh), fill)
        canvas.paste(img, (100, 100))
        return canvas, labels
    elif tmp>0.8:
        negMargin = int(random.uniform(150, 400)) # random background margin
        line = neg[int(random.uniform(1, len(neg)))]
        NegFile = line.split(' ')[0]
        w, h = img.size
        ow, oh = w + negMargin, h + negMargin
        canvas = Image.new('RGB', (ow, oh), fill)
        NegImg = Image.open(os.path.join(negroot, NegFile))
        if NegImg.mode != 'RGB':
            NegImg = NegImg.convert('RGB')
        negw, negh = NegImg.size
        xmin = random.uniform(0, max(10, negw - 50))
        ymin = random.uniform(0, max(10, negh - 50))
        xmax = 0
        ymax = 0
        flag = True
        while flag:
            xmax = random.uniform(xmin, negw)
            ymax = random.uniform(ymin, negh)
            if xmax - xmin > 50 and ymax - ymin > 50:
                flag = False
        NegImg_c = NegImg.crop((int(xmin), int(ymin), int(xmax), int(ymax)))
        NegImg = NegImg_c.resize((ow,oh))
        canvas.paste(NegImg, (0, 0))
        #x_off = int(random.uniform(0,w)) # bug here. wynmew@Dec 26, 2018
        #y_off = int(random.uniform(0,h))
        x_off = int(random.uniform(0, negMargin))
        y_off = int(random.uniform(0, negMargin))
        canvas.paste(img, (x_off, y_off))
        return canvas, labels
    else:
        w, h = img.size
        ow, oh = w+200, h + 200
        canvas = Image.new('RGB', (ow, oh), fill)
        x_off = int(random.uniform(0,200))
        y_off = int(random.uniform(0,200))
        canvas.paste(img, (x_off, y_off))
        return canvas, labels

class ListDatasetV4(data.Dataset):
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
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    #print('c=', c)
                    code_c = self.barcodeDict[c]
                    #print('code_c', code_c)
                    label.append(self.class_to_ind[code_c])
                self.labels.append(torch.LongTensor(label))
            else:
                label = [0]
                self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx].strip()
        #print(fname)
        #print(os.path.join(self.root, fname))
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        labels = self.labels[idx].clone()
        #print(labels)
        #tmp=random.random()
        #margin = 100
        #fill = (0,0,0)
        #if labels[0] != -1:
        if self.transform:
            img, labels = self.transform(img, labels)
        #print(labels)
        #time.sleep(1)
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

class ListDatasetMut(data.Dataset):
    def __init__(self, root, list_file, classes=None, transform=None):
        self.root = root
        self.classes = classes
        self.num_classes = len(classes)
        self.transform = transform
        self.fnames = []
        self.labels = []
        self.class_to_ind = dict(zip(classes, range(len(classes))))
        self.fill=(0,0,0)

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    if c in self.classes:
                        label.append(self.class_to_ind[c])

                self.labels.append(torch.LongTensor(label))
            else:
                label = [0]
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

class ListDatasetV5(data.Dataset):
    def __init__(self, root, list_file, classes=None, transform=None):
        self.root = root
        self.num_classes = len(classes)
        self.transform = transform
        self.fnames = []
        self.labels = []
        self.class_to_ind = dict(zip(classes, range(len(classes))))
        self.fill=(0,0,0)

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    label.append(self.class_to_ind[c])
                    #code_c = self.barcodeDict[c]
                    #label.append(self.class_to_ind[code_c])
                self.labels.append(torch.LongTensor(label))
            else:
                label = [0]
                self.labels.append(torch.LongTensor(label))

    def __getitem__(self, idx):
        fname = self.fnames[idx].strip()
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        labels = self.labels[idx].clone()
        if self.transform:
            img, labels = self.transform(img, labels)
        #print(labels)
        #time.sleep(1)
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

class ListDatasetCLL(data.Dataset):
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
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    code_c = self.barcodeDict[c]
                    label.append(self.class_to_ind[code_c])
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)
            else:
                label = [0]
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx].strip()
        #print(fname)
        #print(os.path.join(self.root, fname))
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.labels[idx]
        if self.transform:
            img, labels = self.transform(img, torch.LongTensor(label))

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

        CandidateLabelSetSize = int(random.uniform(min(5, self.num_classes), min(20, self.num_classes)))
        # always add Neg to CandidateLabelSet
        CandidateLabelSet = label
        CandidateLabelSet.append(0)
        for labelidx in range(CandidateLabelSetSize):
            CandidateLabelSet.append(int(random.uniform(0, self.num_classes - 1)))

        CandidateLabelSet = sorted(list(dict.fromkeys(CandidateLabelSet)), key=int)  # delete duplicated elements and sort in assending manner
        CLS = []
        for iC in range(0, self.num_classes):
            CLS.append([0])

        for iC in CandidateLabelSet:
            CLS[iC][0] = 1

        CLS = torch.ByteTensor(CLS)
        #print(fname, target)
        return img, target, CLS

    def __len__(self):
        return self.num_imgs

class ListDatasetCLLV2(data.Dataset):
    def __init__(self, root, list_file, MinNumCandidateLabel=2, MaxNumCandidateLabel=5, classes=None, barcode_Dict = None, transform=None):
        self.root = root
        self.num_classes = len(classes)
        self.barcodeDict = barcode_Dict
        self.transform = transform
        self.fnames = []
        self.labels = []
        self.class_to_ind = dict(zip(classes, range(len(classes))))
        self.fill=(0,0,0)
        self.MinNumCandidateLabel = MinNumCandidateLabel
        self.MaxNumCandidateLabel = MaxNumCandidateLabel

        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    code_c = self.barcodeDict[c]
                    label.append(self.class_to_ind[code_c])
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)
            else:
                label = [0]
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)

    def __getitem__(self, idx):
        fname = self.fnames[idx].strip()
        #print(fname)
        #print(os.path.join(self.root, fname))
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.labels[idx]
        if self.transform:
            img, labels = self.transform(img, torch.LongTensor(label))

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

        CandidateLabelSetSize = int(random.uniform(min(self.MinNumCandidateLabel, self.num_classes), min(self.MaxNumCandidateLabel, self.num_classes)))
        # always add Neg to CandidateLabelSet
        CandidateLabelSet = label
        CandidateLabelSet.append(0)
        for labelidx in range(CandidateLabelSetSize):
            CandidateLabelSet.append(int(random.uniform(0, self.num_classes - 1)))

        CandidateLabelSet = sorted(list(dict.fromkeys(CandidateLabelSet)), key=int)  # delete duplicated elements and sort in assending manner
        CLS = []
        for iC in range(0, self.num_classes):
            CLS.append([0])

        for iC in CandidateLabelSet:
            CLS[iC][0] = 1

        CLS = torch.ByteTensor(CLS)
        #print(fname, target)
        return img, target, CLS

    def __len__(self):
        return self.num_imgs

class ListDatasetCLLonly1(data.Dataset):
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
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file

        with open(list_file) as f:
            lines = f.readlines()
            self.num_imgs = len(lines)

        for line in lines:
            splited = line.strip().split('/')
            self.fnames.append(line)
            if splited[1] != 'Neg':
                numLabels=len(splited[1].split('_'))
                label = []
                for i in range(numLabels):
                    c = splited[1].split('_')[i]
                    code_c = self.barcodeDict[c]
                    label.append(self.class_to_ind[code_c])
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)
            else:
                label = [0]
                #self.labels.append(torch.LongTensor(label))
                self.labels.append(label)

    def __getitem__(self, idx):
        '''Load image.

        Args:
          idx: (int) image index.

        Returns:
          img: (tensor) image tensor.
          boxes: (tensor) bounding box targets.
          labels: (tensor) class label targets.
        '''
        # Load image and boxes.
        fname = self.fnames[idx].strip()
        #print(fname)
        #print(os.path.join(self.root, fname))
        img = Image.open(os.path.join(self.root, fname))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        label = self.labels[idx]
        if self.transform:
            img, labels = self.transform(img, torch.LongTensor(label))

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

        CandidateLabelSetSize = 1
        # always add Neg to CandidateLabelSet
        CandidateLabelSet = label
        CandidateLabelSet.append(0)
        for labelidx in range(CandidateLabelSetSize):
            CandidateLabelSet.append(int(random.uniform(0, self.num_classes - 1)))

        CandidateLabelSet = sorted(list(dict.fromkeys(CandidateLabelSet)), key=int)  # delete duplicated elements and sort in assending manner
        CLS = []
        for iC in range(0, self.num_classes):
            CLS.append([0])

        for iC in CandidateLabelSet:
            CLS[iC][0] = 1

        CLS = torch.ByteTensor(CLS)
        #print(fname, target)
        return img, target, CLS

    def __len__(self):
        return self.num_imgs

