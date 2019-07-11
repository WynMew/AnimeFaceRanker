import torch
import torch.nn as nn
import numpy as np
import os
from torch.autograd import Variable
from MultiLabelDataLoaderAug import *
from torchvision import datasets, models, transforms
from HRModel import *
import torch.optim as optim
import torch.nn.functional as F
from os.path import exists, join, basename, dirname
from torch.optim import lr_scheduler
import math
# pytorch0.4, pytorch1.0

My_CLASSES = (  # always index 0
'null',
'face'
    )


codeDict={
'del':'null',
'FaceCrop':'face'
}


FilePre='test_HRNet_Face'
torch.cuda.set_device(0)
# Model
print('==> Building model..')
MyClassNum = len(My_CLASSES)
model = MultiLabelClassifierHRNetW30(MyClassNum)
#model = torch.nn.DataParallel(model,device_ids=[0,2,4,6]).cuda()
model.cuda()
m=nn.Sigmoid()
init_lr = 1e-2
img_size=256
optimizer = optim.SGD(model.parameters(), lr= init_lr, momentum=0.5)


def transform_train(img, labels):
    img = random_noise(img)
    img,_ = resize(img, boxes=None, size=(img_size,img_size), random_interpolation=True)
    img = random_flip(img)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, labels

trainset = ListDatasetLite(root='/home/wynmew/data/tachie',
                    list_file='/home/wynmew/data/tachie/faceCropList', # dataset with neg
                    classes = My_CLASSES,
                    barcode_Dict = codeDict,
                    transform=transform_train)

trainBatchSize = 12
trainloader = torch.utils.data.DataLoader(trainset, batch_size=trainBatchSize, shuffle=True, drop_last=True, num_workers=0)

def addlog(x):
    with open(FilePre+"_Log","a+") as outfile:
        outfile.write(x + "\n")

k=1 #OHEM

def train(epoch, model, loss_fn, optimizer, dataloader,log_interval=50):
    model.train()
    train_loss = 0
    for batch_idx, (inputs, cls_targets) in enumerate(dataloader):
        inputs = Variable(inputs.cuda())
        optimizer.zero_grad()
        labelsPre = model(inputs)

        lossCLS = Variable(torch.FloatTensor(1)).zero_()
        lossCLS = lossCLS.cuda()
        # calculate loss, OHEM
        for bID in range(trainBatchSize):  # batch itr
            tmpBatchLoss = Variable(torch.FloatTensor(1)).zero_()
            tmpBatchLoss = tmpBatchLoss.cuda()
            for cID in range(MyClassNum):  # class itr
                sub_loss = loss_fn(m(labelsPre[cID][bID]), Variable(cls_targets[bID][cID].cuda()))
                tmpBatchLoss += sub_loss
            lossCLS = torch.cat((lossCLS, tmpBatchLoss), 0)
        index = torch.topk(lossCLS, int(k * lossCLS.size()[0]))
        valid_lossCLS = lossCLS[index[1]]
        lossCLS=torch.mean(valid_lossCLS)

        loss = lossCLS
        loss.backward()
        optimizer.step()
        #train_loss += loss.data.cpu().numpy()[0] #pytorch0.3
        train_loss += loss.data.cpu().numpy() #pytorch1.0
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
                epoch, batch_idx , len(dataloader), 100. * batch_idx / len(dataloader), loss.data)) #pytorch1.0
            line = "Train Epoch: " + str(epoch) + " " + str(100. * batch_idx / len(dataloader)) + " " + str(loss.data) #pytorch1.0
            #print('Train Epoch: {} [{}/{} ({:.0f}%)]  Loss: {:.6f}'.format(
            #    epoch, batch_idx , len(dataloader), 100. * batch_idx / len(dataloader), loss.data[0])) # pytorch0.3
            #line = "Train Epoch: " + str(epoch) + " " + str(100. * batch_idx / len(dataloader)) + " " + str(loss.data[0])# pytorch0.3
            addlog(line)
    train_loss /= len(dataloader)
    print('Train set: Average loss: {:.4f}'.format(train_loss))
    line = "Train set: Average loss: " + str(train_loss)
    #add(line)
    return train_loss

def adjust_lr(optimizer, epoch, maxepoch, init_lr, power = 0.9):
    lr = init_lr * (1-epoch/maxepoch)**power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

cwd = os.getcwd()
print(cwd)

lossFun = nn.MSELoss()
best_test_loss = float("inf")
print('Starting training...')
start_epoch = 1
end_epoch = 2

for epoch in range(start_epoch, end_epoch + 1):
    train_loss = train(epoch, model, lossFun, optimizer, trainloader,  log_interval=1)
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    lr_now = adjust_lr(optimizer, epoch, end_epoch + 1, init_lr, power=10)
    print(lr_now)
    line = "lr_Now: " + str(lr_now)
    print('Saving..')
    state = {
        'net': model.state_dict(),
        #'net': model.module.state_dict(), #DataParallel
        'epoch': epoch,
    }
    torch.save(state, FilePre+str(epoch)+'.pth.tar')
print('Done!')
