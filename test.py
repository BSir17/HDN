# import torch
# import torch.nn as nn
# import torch.nn.init as init
# from torchsummary import summary
# from torch.autograd import Variable
# from itertools import product as product
# import torch.nn.functional as F
# import numpy as  np
#
#
# # tt=torch.Tensor(512)
# # print(tt)
# # init.constant(tt,5)
# #
# #
# # print(tt)
# import time
#
# tt=torch.Tensor(4,2,64,32)
# tt1=tt.permute(0,2,3,1)
# print(tt1.shape)
#
# for i, j in product(range(3), range(5)):
#     print(i,j)
#
# tt=Variable(torch.Tensor(4,3,16,16))
# print(tt.shape)
# tt=F.upsample(tt,[32,32])
#
# t1=torch.ones([256,32,32])
# t2=torch.ones([256,32,32])*2
# rs=t1*t2
# print(t1*t2)
#
# for i in range(3)[::-1]:
#     print(i)
#
# t0=time.time()
# for i in range(100000):
#     for j in range(1000):
#         ii=i
# t1=time.time()
# print(t1-t0)
#
# a=np.arange(16).reshape(4,-1)
# try:
#     dets = np.row_stack((dets, a))
# except:
#     dets=np.arange(16).reshape(4,-1)
#     print("herere?")
# print("???")
# exit(0)
#
#
# vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
#            512, 512, 512, 'M']
#
# extras_cfg = [256, 'S', 512, 128, 'S', 256]
#
# def vgg1(cfg, i, batch_norm=False):
#     layers = []
#     in_channels = i
#     for v in cfg:
#         if v == 'M':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#         elif v == 'C':
#             layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
#         else:
#             conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
#             if batch_norm:
#                 layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]            #NN.BARCHNORM2D num_features:特征的数量
#             else:
#                 layers += [conv2d, nn.ReLU(inplace=True)]
#             in_channels = v
#     conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
#     conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
#     layers += [conv6,
#                nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
#
#     # summary(layers,(3,244,244))
#     return layers
#
#
# def add_extras1(cfg, i, batch_norm=False):
#     # Extra layers added to VGG for feature scaling
#     layers = []
#     in_channels = i
#     flag = False
#     for k, v in enumerate(cfg):
#         if in_channels != 'S':
#             if v == 'S':
#                 layers += [nn.Conv2d(in_channels, cfg[k + 1],
#                                      kernel_size=(1, 3)[flag], stride=2, padding=1)]
#             else:
#                 layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
#             flag = not flag
#         in_channels = v
#     return layers
#
# class vggg(nn.Module):
#     def __init__(self):
#         super(vggg,self).__init__()
#         self.vgg1 = nn.ModuleList(vgg1(vgg_cfg,3))
#         self.add_extras1=add_extras1(extras_cfg,1024)
#     def forward(self,x):
#         for k in range(len(self.vgg1)):
#             x = self.vgg1[k](x)
#         for k, v in enumerate(self.add_extras1):
#             x = F.relu(v(x), inplace=True)
#         return x
#
# use_cuda = torch.cuda.is_available()
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# if use_cuda:
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')
# model = vggg()
# summary(model, (3,640,640))
#
#
#

import numpy as np

s=[3,4,2,1,5]
s=np.array(s)
print(s[[2,3,4]])
s=[0]*5
print(s)