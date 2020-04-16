#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Function
from torch.autograd import Variable



class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))       #torch.size(self.n_channels).加入nn.Parameter则要求梯度
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)                       #the second parameter:the value to fill the tensor with

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps    #归一化的范数.是所有参数的平方和然后再开根号
        #x /= norm
        x = torch.div(x,norm)                                       #归一化
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x       #归一化后乘上scale
        return out


        