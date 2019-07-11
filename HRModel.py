from __future__ import print_function, division
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import os
import pprint
import shutil
import sys
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from lib.config import config
from lib.config import update_config
#pytorch 0.4.1, pytorch 1.0.1
import argparse
import torch.nn.functional as F

BN_MOMENTUM = 0.1

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(False)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i],
                                       momentum=BN_MOMENTUM),
                        nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.ReLU(False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        super(HighResolutionNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        # Classification Head
        self.incre_modules, self.downsamp_modules, \
        self.final_layer = self._make_head(pre_stage_channels)

        #self.classifier = nn.Linear(2048, 1000)

    def _make_head(self, pre_stage_channels):
        head_block = Bottleneck
        head_channels = [32, 64, 128, 256]

        # Increasing the #channels on each resolution 
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        incre_modules = []
        for i, channels in enumerate(pre_stage_channels):
            incre_module = self._make_layer(head_block,
                                            channels,
                                            head_channels[i],
                                            1,
                                            stride=1)
            incre_modules.append(incre_module)
        incre_modules = nn.ModuleList(incre_modules)

        # downsampling modules
        downsamp_modules = []
        for i in range(len(pre_stage_channels) - 1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i + 1] * head_block.expansion

            downsamp_module = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            )

            downsamp_modules.append(downsamp_module)
        downsamp_modules = nn.ModuleList(downsamp_modules)

        final_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=head_channels[3] * head_block.expansion,
                out_channels=2048,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.BatchNorm2d(2048, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )

        return incre_modules, downsamp_modules, final_layer

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  3,
                                  1,
                                  1,
                                  bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(
                            inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # Classification Head
        y = self.incre_modules[0](y_list[0])
        for i in range(len(self.downsamp_modules)):
            y = self.incre_modules[i + 1](y_list[i + 1]) + \
                self.downsamp_modules[i](y)

        y = self.final_layer(y)

        y = F.avg_pool2d(y, kernel_size=y.size()
        [2:]).view(y.size(0), -1)

        #y = self.classifier(y)

        return y

    def init_weights(self, pretrained='', ):
        #logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            #logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


def get_cls_net(config, **kwargs):
    model = HighResolutionNet(config, **kwargs)
    model.init_weights()
    return model

class FeatureExtractionHRNetW30(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractionHRNetW30, self).__init__()
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            # required=True,
                            type=str,
                            default='cls_hrnet_w30_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                            )
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--testModel',
                            help='testModel',
                            type=str,
                            default='hrnetv2_w30_imagenet_pretrained.pth')
        args = parser.parse_args()
        update_config(config, args)

        self.HRNet = get_cls_net(config)
        self.HRNet.init_weights(pretrained = 'hrnetv2_w30_imagenet_pretrained.pth')

    def forward(self, image_batch):
        return self.HRNet(image_batch)

class HRNetW30classifier(nn.Module):
    def __init__(self,num_classes):
        super(HRNetW30classifier, self).__init__()
        self.num_classes = num_classes
        for index in enumerate(range(0, self.num_classes)):
            setattr(self, "FullyConnectedLayer_" + str(index[0]),
                    nn.Linear(2048, 1),
                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        outs = list()
        for index in enumerate(range(0, self.num_classes)):
            fun = eval("self.FullyConnectedLayer_" + str(index[0]))
            out = fun(x)
            outs.append(out)
        return outs

class MultiLabelClassifierHRNetW30(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifierHRNetW30, self).__init__()
        self.FeatureExtraction = FeatureExtractionHRNetW30()
        self.classifier = HRNetW30classifier(num_classes)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        preLabels= self.classifier(feature)
        return preLabels # list of predicatons

#dumpinput = torch.rand((1, 3, config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]))

class FeatureExtractionHRNetW32(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractionHRNetW32, self).__init__()
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            # required=True,
                            type=str,
                            default='cls_hrnet_w32_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                            )
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--testModel',
                            help='testModel',
                            type=str,
                            default='hrnetv2_w32_imagenet_pretrained.pth')
        args = parser.parse_args()
        update_config(config, args)

        self.HRNet = get_cls_net(config)
        self.HRNet.init_weights(pretrained = 'hrnetv2_w32_imagenet_pretrained.pth')

    def forward(self, image_batch):
        return self.HRNet(image_batch)

class HRNetW32classifier(nn.Module):
    def __init__(self,num_classes):
        super(HRNetW32classifier, self).__init__()
        self.num_classes = num_classes
        for index in enumerate(range(0, self.num_classes)):
            setattr(self, "FullyConnectedLayer_" + str(index[0]),
                    nn.Linear(2048, 1),
                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        outs = list()
        for index in enumerate(range(0, self.num_classes)):
            fun = eval("self.FullyConnectedLayer_" + str(index[0]))
            out = fun(x)
            outs.append(out)
        return outs

class MultiLabelClassifierHRNetW32(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifierHRNetW32, self).__init__()
        self.FeatureExtraction = FeatureExtractionHRNetW32()
        self.classifier = HRNetW32classifier(num_classes)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        preLabels= self.classifier(feature)
        return preLabels # list of predicatons

class FeatureExtractionHRNetW40(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractionHRNetW40, self).__init__()
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            # required=True,
                            type=str,
                            default='cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                            )
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--testModel',
                            help='testModel',
                            type=str,
                            default='hrnetv2_w40_imagenet_pretrained.pth')
        args = parser.parse_args()
        update_config(config, args)

        self.HRNet = get_cls_net(config)
        self.HRNet.init_weights(pretrained = 'hrnetv2_w40_imagenet_pretrained.pth')

    def forward(self, image_batch):
        return self.HRNet(image_batch)

class HRNetW40classifier(nn.Module):
    def __init__(self,num_classes):
        super(HRNetW40classifier, self).__init__()
        self.num_classes = num_classes
        for index in enumerate(range(0, self.num_classes)):
            setattr(self, "FullyConnectedLayer_" + str(index[0]),
                    nn.Linear(2048, 1),
                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        outs = list()
        for index in enumerate(range(0, self.num_classes)):
            fun = eval("self.FullyConnectedLayer_" + str(index[0]))
            out = fun(x)
            outs.append(out)
        return outs

class MultiLabelClassifierHRNetW40(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifierHRNetW40, self).__init__()
        self.FeatureExtraction = FeatureExtractionHRNetW40()
        self.classifier = HRNetW40classifier(num_classes)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        preLabels= self.classifier(feature)
        return preLabels # list of predicatons

class FeatureExtractionHRNetW44(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractionHRNetW44, self).__init__()
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            # required=True,
                            type=str,
                            default='cls_hrnet_w44_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                            )
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--testModel',
                            help='testModel',
                            type=str,
                            default='hrnetv2_w44_imagenet_pretrained.pth')
        args = parser.parse_args()
        update_config(config, args)

        self.HRNet = get_cls_net(config)
        self.HRNet.init_weights(pretrained = 'hrnetv2_w44_imagenet_pretrained.pth')

    def forward(self, image_batch):
        return self.HRNet(image_batch)

class HRNetW44classifier(nn.Module):
    def __init__(self,num_classes):
        super(HRNetW44classifier, self).__init__()
        self.num_classes = num_classes
        for index in enumerate(range(0, self.num_classes)):
            setattr(self, "FullyConnectedLayer_" + str(index[0]),
                    nn.Linear(2048, 1),
                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        outs = list()
        for index in enumerate(range(0, self.num_classes)):
            fun = eval("self.FullyConnectedLayer_" + str(index[0]))
            out = fun(x)
            outs.append(out)
        return outs

class MultiLabelClassifierHRNetW44(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifierHRNetW44, self).__init__()
        self.FeatureExtraction = FeatureExtractionHRNetW44()
        self.classifier = HRNetW44classifier(num_classes)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        preLabels= self.classifier(feature)
        return preLabels # list of predicatons

class FeatureExtractionHRNetW48(torch.nn.Module):
    def __init__(self):
        super(FeatureExtractionHRNetW48, self).__init__()
        parser = argparse.ArgumentParser(description='Train classification network')
        parser.add_argument('--cfg',
                            help='experiment configure file name',
                            # required=True,
                            type=str,
                            default='cls_hrnet_w48_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
                            )
        parser.add_argument('--modelDir',
                            help='model directory',
                            type=str,
                            default='')
        parser.add_argument('--logDir',
                            help='log directory',
                            type=str,
                            default='')
        parser.add_argument('--dataDir',
                            help='data directory',
                            type=str,
                            default='')
        parser.add_argument('--testModel',
                            help='testModel',
                            type=str,
                            default='hrnetv2_w48_imagenet_pretrained.pth')
        args = parser.parse_args()
        update_config(config, args)

        self.HRNet = get_cls_net(config)
        self.HRNet.init_weights(pretrained = 'hrnetv2_w48_imagenet_pretrained.pth')

    def forward(self, image_batch):
        return self.HRNet(image_batch)

class HRNetW48classifier(nn.Module):
    def __init__(self,num_classes):
        super(HRNetW48classifier, self).__init__()
        self.num_classes = num_classes
        for index in enumerate(range(0, self.num_classes)):
            setattr(self, "FullyConnectedLayer_" + str(index[0]),
                    nn.Linear(2048, 1),
                    )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        outs = list()
        for index in enumerate(range(0, self.num_classes)):
            fun = eval("self.FullyConnectedLayer_" + str(index[0]))
            out = fun(x)
            outs.append(out)
        return outs

class MultiLabelClassifierHRNetW48(nn.Module):
    def __init__(self, num_classes):
        super(MultiLabelClassifierHRNetW48, self).__init__()
        self.FeatureExtraction = FeatureExtractionHRNetW48()
        self.classifier = HRNetW48classifier(num_classes)

    def forward(self, img):
        # do feature extraction
        feature = self.FeatureExtraction(img)
        preLabels= self.classifier(feature)
        return preLabels # list of predicatons

#import random
#date
#random.sample(range(1,128),3)
