import numpy as np
from MultiLabelDataLoaderAug import *
from torchvision import datasets, models, transforms
from HRModel import *
import torch.nn.functional as F
import linecache as lc
import cv2
from torch.autograd import Variable
import os.path as osp
import json
from collections import OrderedDict
import shutil

My_CLASSES = (  # always index 0
'null',
'face'
    )


codeDict={
'del':'null',
'FaceCrop':'face'
}

class_to_ind = dict(zip(My_CLASSES, range(len(My_CLASSES))))
weightfile = 'examples/test_HRNet_Face1.pth.tar'

margin = 0
#margin = 80
# Model
print('==> Building model..')
MyClassNum = len(My_CLASSES)

model = MultiLabelClassifierHRNetW30(MyClassNum)

model.load_state_dict(torch.load(weightfile,map_location=lambda storage, loc:storage)['net'])
model.eval()
model.cuda()

img_size=256
#img_size=224

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

counter = 0
s=nn.Sigmoid()
sm=nn.Softmax()
margin=100
ScoreTH=-2


idx=[]

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

DataFile = '/home/wynmew/data/downloads/danbooru2018/origCropList20190708'


for line in open(osp.join(DataFile)):
    idx.append((line.strip()))


for i in range(len(idx)):
    imgpath= idx[i]
    imgCV2 = cv2.imread(imgpath)
    imgPre = cv2.cvtColor(imgCV2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(imgPre)
    img_pil_r = img_pil.resize((img_size, img_size))
    inputs = preprocess(img_pil_r)
    inputs.unsqueeze_(0)
    LabelPre = model(Variable(inputs.cuda(), volatile=True))
    score = s(LabelPre[1].cpu()).item()
    print(imgpath, score)
    #if score <0.8:
    if score < 0.6:
        shutil.copy2(imgpath, '/home/wynmew/data/downloads/danbooru2018/origCropRankerDel/')
    else:
        shutil.copy2(imgpath, '/home/wynmew/data/downloads/danbooru2018/origCropRankerSel/')