# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 14:32:01 2022

@author: adywi
"""

from tqdm import tqdm
from pathlib import Path
from glob import glob
import math, time, os, re
import numpy as np
import pickle
import torch
from torch.optim.optimizer import Optimizer
from torch.autograd import Variable


import torch.nn as nn


def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()

        if groups != 1 or (base_width != 64 and base_width != 32):
            raise ValueError('BasicBlock only supports groups=1 and base_width=64 or base_width=32')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        # TODO: why 64?
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

#ResNet
class Feature_extractor(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, norm_layer=None):
        super(Feature_extractor, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self._norm_layer = norm_layer

        # TODO: why 64?
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv3d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        #self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def _resnet(arch, inplanes, planes, pretrained, progress, **kwargs):
    model = ResNet(inplanes, planes, **kwargs)
    if pretrained:
        raise ValueError("Pretrained not implemented")
    return model


def _s3resnet(arch, inplanes, planes, pretrained, progress, smaller, devices=None, **kwargs):
    if devices:
        model = S3ResNetMultiGPU(inplanes, planes, devices, **kwargs)
    else:
        if smaller:
            print("SMALLER ResNet18")

            model = S3ResNetSmall(inplanes, planes, **kwargs)
        else:
            print("STANDARD ResNet18")
            model = S3ResNet(inplanes, planes, **kwargs)

    if pretrained:
        raise ValueError("Pretrained not implemented")
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def s3resnet18(pretrained=False, progress=True, smaller=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _s3resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress, smaller,
                     **kwargs)


class S3ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7, zero_init_residual=False, groups=1, width_per_group=32,
                 replace_stride_with_dilation=None, norm_layer=None, in_channel=3, final_n_channel=512, no_fc=False):
        super(S3ResNet, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self._norm_layer = norm_layer
        self.no_fc = no_fc
    
        # TODO: why 64?
        self.inplanes = 64
        self.dilation = 1
        self.final_n_channel = final_n_channel
        replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv3d(in_channel, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        #         self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, final_n_channel, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        if not no_fc:
            self.fc = nn.Linear(final_n_channel * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.no_fc:
            return x
        else:
            out = self.fc(x)

        return out, x

        
        
        
class S3ConvXFCResnet(nn.Module):
    def __init__(self, in_channel, num_classes,drop_out=False):
        super(S3ConvXFCResnet, self).__init__()
        self.n_classes=num_classes
        
        self.in_channel, self.num_classes = in_channel, num_classes

        # self._time_conv = nn.Conv3d(self.in_channel, 3, 1, stride=1, padding=0, bias=False)
        # nn.init.kaiming_normal_(self._time_conv.weight, mode='fan_out', nonlinearity='relu')
        
        # first in_channel is number of frames
        self.time_conv = nn.Sequential(
            nn.Conv3d(self.in_channel, 3, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(3),
            nn.ReLU(inplace=True)
        )
        
        
            
        self.s3resnet = s3resnet18(num_classes=self.n_classes, no_fc=True)
        
        self.Feature_extractor = nn.Sequential(
            self.time_conv,
            self.s3resnet
        )

        # self.convs = nn.DataParallel(self.convs)
        
        
        
        final_n_channel=self.Feature_extractor[-1].inplanes
        #block.expansion #TODO initial expension, but block is hidden in the creation
        expansion=1
        self.classifier = nn.Sequential(
            nn.Linear(final_n_channel * expansion, num_classes),
            nn.LogSoftmax(dim=1))
            
            

        # weights initialisation    
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
        
        
    def forward(self, x, return_emb=False):
        out = self.Feature_extractor(x)
        x = self.classifier(out)
        if return_emb:
            return out, x
        else:
            return x
    
    def FT(self,FT_params):
        """
        
        Wrapper, prepare the model to fine_tune
        Update current object accordingly
        
        
        STEP 1 LOAD the MODEL
        STEP 2 adapt output or classifier for transfert learning
        STEP 3 freeze desired layers


        Parameters for fine tunning the model
        ----------
        
        parameters : Dictionnary 
            {"path_weights": path of pretrained weights,
             "reset": "all", "last_layer",
             "n_output":int /output dim of the last layer
             "freeze_type":"all","feature_extractor","progressive",
             "drop_out":Bool / usefull only if more than one layer is changed
             }
            
        
        Returns
        model : torch.nn model 
            pretrained model for transfert learning
            
            ready to train
            
            
        
        """
        self.load_model(FT_params["path_weights"])
        self.FT_adapt_classifier(FT_params["reset"],FT_params["n_output"],FT_params["drop_out"])
        self.FT_freeze_params(FT_params["freeze_type"])
        return self
    
    
    def FT_test(self,FT_params):
        """
        
        Wrapper, prepare the model to fine_tune
        Update current object accordingly
        
        
        STEP 1 LOAD the MODEL
        STEP 2 adapt output or classifier for transfert learning
        STEP 3 freeze desired layers


        Parameters for fine tunning the model
        ----------
        
        parameters : Dictionnary 
            {"path_weights": path of pretrained weights,
             "reset": "all", "last_layer",
             "n_output":int /output dim of the last layer
             "freeze_type":"all","feature_extractor","progressive",
             "drop_out":Bool / usefull only if more than one layer is changed
             }
            
        
        Returns
        model : torch.nn model 
            pretrained model for transfert learning
            
            ready to train
            
            
        
        """
        #self.load_model(FT_params["path_weights"])
            
        self.FT_adapt_classifier(FT_params["reset"],FT_params["n_output"],FT_params["drop_out"])
        self.FT_freeze_params(FT_params["freeze_type"])
        return self
    
    def load_model(self,path):
        """
        The model is modified in structure, since we slice a model
        into submodules. we iterate over the layers to get the right initialisation

        """
        pre_trained_model = torch.load(path,
                                         map_location=torch.device('cpu'))['state_dict']#TODO change cpu device if neede
        
        self.load_state_dict(pre_trained_model)
        
            
    def FT_adapt_classifier(self,reset, n_output, drop_out=False):
        """
        Modify the classifier for fine tune the model
        #replace both or just the last layer then initialize_weights
        
        reset:
        all: all layers are changed
        last_layer : last layer is changed
        None: No change
              
        """
        self.n_classes=n_output
        
        if reset=="all":
            final_n_channel=self.Feature_extractor[-1].inplanes
            #block.expansion #TODO initial expension, but block is hidden in the creation
            expansion=1
            if drop_out:
                self.fc = nn.Linear(final_n_channel * expansion, n_output)
                self.classifier = nn.Sequential(
                    nn.Linear(final_n_channel * expansion, n_output),
                    nn.LogSoftmax(dim=1))
                
                
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(final_n_channel * expansion, n_output),
                    nn.LogSoftmax(dim=1))
            # weights initialisation    
            for m in self.classifier.modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()
                    
        elif reset=="last_layer":#TODO
            self.classifier[-2]=nn.Linear(self.classifier[-2].in_features,
                                                    n_output)
            # weights initialisation
            self.classifier[-2].weight.data.normal_(0, 0.01)
            self.classifier[-2].bias.data.zero_()
        else:
            pass
        
        
        
    def FT_freeze_params(self,freeze_type):
        """
        freeze layers to prevent updating weights
        all: all layers but the last one
        feature_extractor : only the layers of the Feature extractor are frozen
        progressive:learning rate decaying with deepness#TODO
        None: No freeze

        """
        
        if freeze_type=="all":
            for param in self.parameters():#freeze all layers
                param.requires_grad = False
            self.classifier[-2].weight.requires_grad = True #unfreeze last layer
            self.classifier[-2].bias.requires_grad = True #unfreeze last layer
            
        elif freeze_type=="feature_extractor":
            #freeze only
            for param in self.Feature_extractor.parameters():
                param.requires_grad = False
        elif freeze_type=="progressive":
            pass
    
"""    
model=S3ConvXFCResnet(27,8)

import os
path_records = "Records/"
path_records = os.path.abspath(path_records)
weights_version='Weights_models/checkpoint_24.pth.tar'
path_weights=os.path.join(path_records, weights_version)

dic={#"path_weights": path_weights,
 "reset": "last_layer",
 "n_output":8,
 "freeze_type":"feature_extractor",
 "drop_out":False
 }

model.FT_test(dic)

for p in model.Feature_extractor.parameters():
    print(p.requires_grad)
for p in model.classifier.parameters():
    print(p.requires_grad)
"""
""""""