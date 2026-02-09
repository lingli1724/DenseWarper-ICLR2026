from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F
from deform_conv import DeformConv

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

class PoseHighResolutionNet(nn.Module):
    def __init__(self,njoints):
        super(PoseHighResolutionNet, self).__init__()
        inner_ch=128
        self.njoints=njoints
        self.offset_feats = self._compute_chain_of_basic_blocks(njoints,inner_ch,20)
        ### Offsets
        k=3
        self.offsets1 = self._single_conv(inner_ch, k, k, 3, njoints)
        self.offsets2 = self._single_conv(inner_ch, k, k, 6, njoints)
        self.offsets3 = self._single_conv(inner_ch, k, k, 12, njoints)
        self.offsets4 = self._single_conv(inner_ch, k, k, 18, njoints)
        self.offsets5 = self._single_conv(inner_ch, k, k, 24, njoints)
        '''
        print(self.offsets1.weight.requires_grad)
        print(self.offsets2.weight.requires_grad)
        print(self.offsets3.weight.requires_grad)
        print(self.offsets4.weight.requires_grad)
        print(self.offsets5.weight.requires_grad)
        '''
        ### Deform conv
        self.deform_conv1 = self._deform_conv(njoints, k, k, 3, njoints)
        self.deform_conv2 = self._deform_conv(njoints, k, k, 6, njoints)
        self.deform_conv3 = self._deform_conv(njoints, k, k, 12, njoints)
        self.deform_conv4 = self._deform_conv(njoints, k, k, 18, njoints)
        self.deform_conv5 = self._deform_conv(njoints, k, k, 24, njoints)
        '''
        print(self.deform_conv1.weight.requires_grad)
        print(self.deform_conv2.weight.requires_grad)
        print(self.deform_conv3.weight.requires_grad)
        print(self.deform_conv4.weight.requires_grad)
        print(self.deform_conv5.weight.requires_grad)
        '''

    def _deform_conv(self, nc, kh, kw, dd, dg):
       conv_offset2d = DeformConv(
           nc,
           nc, (kh, kw),
           stride=1,
           padding=int(kh/2)*dd,
           dilation=dd,
           deformable_groups=dg)
       return conv_offset2d

    def _single_conv(self, nc, kh, kw, dd, dg):
       conv = nn.Conv2d(
           nc,
           dg * 2 * kh * kw,
           kernel_size=(3, 3),
           stride=(1, 1),
           dilation=(dd, dd),
           padding=(1*dd, 1*dd),
           bias=False)
       return conv
    
    def _compute_chain_of_basic_blocks(self, nc, ic, b):
        num_blocks = b
        block = BasicBlock
        in_ch = ic
        out_ch = ic
        stride = 1

        downsample = nn.Sequential(
            nn.Conv2d(
                nc,
                in_ch,
                kernel_size=1, stride=stride, bias=False
            ),
            nn.BatchNorm2d(
                in_ch,
                momentum=BN_MOMENTUM
            ),
        )

        layers = []
        layers.append(
            block(
                nc,
                out_ch,
                stride,
                downsample
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                block(
                    in_ch,
                    out_ch
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, y):#(batch,njoints,h,w)
        diff_x=y-x
        sup_x_cuda=x
        off_feats_cuda = self.offset_feats(diff_x)
        
        off1 = self.offsets1(off_feats_cuda)
        warped_x1 = self.deform_conv1(sup_x_cuda, off1)

        off2 = self.offsets2(off_feats_cuda)
        warped_x2 = self.deform_conv2(sup_x_cuda, off2)

        off3 = self.offsets3(off_feats_cuda)
        warped_x3 = self.deform_conv3(sup_x_cuda, off3)

        off4 = self.offsets4(off_feats_cuda)
        warped_x4 = self.deform_conv4(sup_x_cuda, off4)

        off5 = self.offsets5(off_feats_cuda)
        warped_x5 = self.deform_conv5(sup_x_cuda, off5)

        warped_x = 0.2 * (warped_x1 + warped_x2 + warped_x3 + warped_x4 + warped_x5)
        return warped_x
        #return y
