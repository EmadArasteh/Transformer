#!/usr/bin/env python
# coding: utf-8



import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias)
        
        self.__padding = (kernel_size - 1) * dilation
        
    def forward(self, input):
        # print(self)
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self,in_channels=1,embedding_size=256,k=5):
        super(context_embedding,self).__init__()
        # print(self)
        # print(in_channels)
        # print(embedding_size)
        # print(k)
        in_channels=13
        self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)
        # self.causal_convolution = CausalConv1d(in_channels,embedding_size,kernel_size=k)


    def forward(self,x):
        x = self.causal_convolution(x)
        # print(x.shape)
        return torch.tanh(x)

