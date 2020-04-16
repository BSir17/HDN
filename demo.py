#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import cv2
import time
import numpy as np
from PIL import Image

from data.config import cfg
from s3fdx import build_s3fd
from torch.autograd import Variable
from utils.augmentations import to_chw_bgr


parser = argparse.ArgumentParser(description='s3df demo')
parser.add_argument('--save_dir', type=str, default='tmp/smart_test/',
                    help='Directory for detect result')
parser.add_argument('--model', type=str,
                    default='weights/after_wider/sfd_head.pth', help='trained model')
parser.add_argument('--thresh', default=0.6, type=float,
                    help='Final confidence threshold')
args = parser.parse_args()



if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()



if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
    #img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1700 * 1200 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                      fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)
    #image = cv2.resize(img, (640, 640))
    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0),requires_grad=False)
    print(x.shape)
    if use_cuda:
        x = x.cuda()
    t1 = time.time()
    y = net(x)
    detections = y.data         #torch.size:[1,2,750,5]
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])          #original img shape

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)


    # for i in range(detections.size(1)):
    #     j = 0
    #     while detections[0, i, j, 0] >= thresh:             #5iterms the first is the score
    #         score = detections[0, i, j, 0]
    #         pt = (detections[0, i, j, 1:] * scale).cpu().numpy()        #left up x,left up y ,right bottom x ,right bottom y
    #         left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
    #         j += 1
    #         cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
    #         conf = "{:.3f}".format(score)
    #         point = (int(left_up[0]), int(left_up[1] - 5))
    #         #cv2.putText(img, conf, point, cv2.FONT_HERSHEY_COMPLEX,
    #         #            0.6, (0, 255, 0), 1)

    dclone=[]
    for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= thresh:             #5iterms the first is the score
                score = detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()        #left up x,left up y ,right bottom x ,right bottom y
                dclone.append([pt[0], pt[1],pt[2], pt[3],score])
                j += 1
    dclone=np.array(dclone)
    dclone = dclone[[my_nms(np.array(dclone), 0.3)]]

    for i in range(dclone.shape[0]):
        pt=dclone[i]
        left_up, right_bottom = (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3]))
        cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
    t2 = time.time()
    print('detect:{} timer:{}'.format(img_path, t2 - t1))
    cv2.imwrite(os.path.join(args.save_dir, os.path.basename(img_path)), img)

def my_nms(dects,thresh):
    x1 = dects[:, 0]
    y1 = dects[:, 1]
    x2 = dects[:, 2]
    y2 = dects[:, 3]
    scores = dects[:, 4]
    index = scores.argsort()[::-1]
    keep=[]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    ndets=index.size
    suppressed = [0]*ndets
    for _i in range(index.size):
        i = index[_i]

        if suppressed[i] == 1:
            continue
        keep.append(i)

        ix1 = x1[i]
        iy1 = y1[i]
        ix2 = x2[i]
        iy2 = y2[i]

        iarea = areas[i]

        for _j in range(_i + 1, ndets):
            j = index[_j]
            if suppressed[j] == 1:
                continue
            xx1 = max(ix1, x1[j])
            yy1 = max(iy1, y1[j])
            xx2 = min(ix2, x2[j])
            yy2 = min(iy2, y2[j])

            w = max(0.0, xx2 - xx1 + 1)
            h = max(0.0, yy2 - yy1 + 1)

            overlap = w * h
            ious = overlap / (iarea + areas[j] - overlap)
            if ious > thresh:
                suppressed[j] = 1

    return keep







if __name__ == '__main__':
    img_path = './img'

    img_path = '/home/fan/PycharmProjects/Pyramidbox.pytorch-master/data/smart_pics_ori/xiao/val'

    img_list = [os.path.join(img_path, x)
                for x in os.listdir(img_path) if x.endswith('jpg')]


    for path in img_list:
        net = build_s3fd('test', cfg.NUM_CLASSES)
        net.load_state_dict(torch.load(args.model))
        # net.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))

        net.eval()

        if use_cuda:
            net.cuda()
            cudnn.benckmark = True


        detect(net, path, args.thresh)
        torch.cuda.empty_cache()

# if __name__ == '__main__':
#     net = build_s3fd('test', cfg.NUM_CLASSES)
#     net.load_state_dict(torch.load(args.model))
#     # net.load_state_dict(torch.load(args.model,map_location=torch.device('cpu')))
#
#     net.eval()
#
#     if use_cuda:
#         net.cuda()
#         cudnn.benckmark = True
#
#
#
#     img_path = './img'
#     img_list = [os.path.join(img_path, x)
#                 for x in os.listdir(img_path) if x.endswith('jpg')]

    # for path in img_list:
    #     detect(net, path, args.thresh)
    #     torch.cuda.empty_cache()
