#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import torch
from itertools import product as product
import math


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
以中心偏移形式计算每个源要素地图的priorbox坐标
为每个特征图计算先验框的坐标(中心偏离的形式)
    """

    def __init__(self, input_size, feature_maps,cfg):
        super(PriorBox, self).__init__()
        self.imh = input_size[0]
        self.imw = input_size[1]

        # number of priors for feature map location (either 4 or 6)
        self.variance = cfg.VARIANCE or [0.1]
        #self.feature_maps = cfg.FEATURE_MAPS
        self.min_sizes = cfg.ANCHOR_SIZES           #anchor scale从16到512.
        self.steps = cfg.STEPS                      #即论文里的stride,从4到128
        self.clip = cfg.CLIP
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')
        self.feature_maps = feature_maps


    def forward(self):
        mean = []
        for k in range(len(self.feature_maps)):
            feath = self.feature_maps[k][0]
            featw = self.feature_maps[k][1]
            for i, j in product(range(feath), range(featw)):
                f_kw = self.imw / self.steps[k]                     #原始图像输入大小/步长
                f_kh = self.imh / self.steps[k]                     #原始图像输入大小/步长

                cx = (j + 0.5) / f_kw
                cy = (i + 0.5) / f_kh

                s_kw = self.min_sizes[k] / self.imw                 #该层的anchor大小除以原始图像输入大小
                s_kh = self.min_sizes[k] / self.imh

                mean += [cx, cy, s_kw, s_kh]                        #特征图对应原图中的位置(0.XX的形式)，以及当前anchor/原图size

        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output                                    #return size:N*4,N是6层feature_maps里的点数之和


if __name__ == '__main__':
    from data.config import cfg
    p = PriorBox([640, 640], cfg)
    out = p.forward()
    print(out.size())
