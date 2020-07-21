from __future__ import division
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import pickle as pkl
import random
import argparse
import imageio
import math
import string
import queue

# need to be consistent
# classes, weights, cfg


def get_test_input(input_dim, CUDA):
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :] / 255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    if CUDA:
        img_ = img_.cuda()

    return img_


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()    #BGR转RGB，在转置 w h c
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    img_ = img_.numpy()  #add by xlx
    return img_, orig_im, dim

# def prep_image(orig_im, inp_dim):
#     dim = orig_im.shape[1], orig_im.shape[0]
#     img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
#     img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy() #(3 608 608)
#     img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
#     img_ = img_.numpy()
#     return img_, orig_im, dim


def write(x, img, classes, colors):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2_new = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2_new, color, -1)

    # caldulate distance z
    width = img.shape[1]
    height = img.shape[0]
    pi = 3.1415926
    f = 0.04  # focal length
    d1 = 2.0  # the distance between the closest place the camera can shot and the camera
    H = 1.5  # the height of the camera
    scale = 0.00003  # Pixel size
    beta1 = math.atan(d1 / H)
    theta = math.atan(height * scale / (2 * f))
    y2 = c2[1].item()
    y = height - y2

    if y < 0.5 * height:
        alpha = theta - math.atan((0.5 * height * scale - y * scale) / f)
    elif y > 0.5 * height:
        alpha = theta + math.atan((y * scale - 0.5 * height * scale) / f)
    else:
        alpha = theta
    if (c2[1] - c1[1]) != 0:
        b1 = (y * scale * math.sqrt(H * H + d1 * d1) / math.sqrt(f * f + (height * scale / 2) * (height * scale / 2)))
        d2 = (b1 * math.sin(pi / 2 - theta + alpha)) / math.sin(pi / 2 - alpha - beta1)
        distance_z = round(d1 + d2, 2)
    else:
        distance_z = 0

    # calculate distance x
    gama = 81.5  # horizontal angle of camera
    W = 3  # 1/2 width of the closest place that the camera can shot
    x1 = c1[0].item()
    x2 = c2[0].item()
    x_kernel = 1 / 2 * (x1 + x2)
    distance_x = round(2 * pi * 65 / 360 * distance_z / width * (x_kernel - 1 / 2 * width), 2)

    labdis = label + ' ' + str(distance_z) + ' ' + str(distance_x)

    cv2.putText(img, labdis, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    result = [distance_z, distance_x, x[1].item(), x[2].item(), x[3].item(), x[4].item(), label]

    return img, result


def cal_v(x_pre, x_past, delt_t, img):
    """
    calculate speed of objects

    """
    v_z = 0.0
    v_x = 0.0
    TTC = 0.0
    v_zTTC = '0'
    v_xTTC = '0'
    distance_z_pre = x_pre[0]
    distance_x_pre = x_pre[1]
    c1_prex = x_pre[2]
    c1_prey = x_pre[3]
    c2_prex = x_pre[4]
    c2_prey = x_pre[5]
    c_prex = (c1_prex + c2_prex) / 2
    c_prey = (c1_prey + c2_prey) / 2

    for i in range(len(x_past)):
        distance_z_past = x_past[i][0]
        distance_x_past = x_past[i][1]
        c1_pastx = x_past[i][2]
        c1_pasty = x_past[i][3]
        c2_pastx = x_past[i][4]
        c2_pasty = x_past[i][5]
        c_pastx = (c1_pastx + c2_pastx) / 2
        c_pasty = (c1_pasty + c2_pasty) / 2

        if (abs(c_pastx - c_prex) <= abs(c2_pastx - c1_pastx) / 2) and (
                abs(c_pasty - c_prey) <= (c2_pasty - c1_pasty) / 2):
            deltdistance_z = distance_z_pre - distance_z_past
            v_z = round(deltdistance_z / delt_t, 2)

            deltdistance_x = distance_x_pre - distance_x_past
            v_x = round(deltdistance_x / delt_t, 2)

            if v_z != 0:
                TTC_z = round(deltdistance_z / v_z, 2)
                # v_zTTC = str(v_z) + ' ' + str(TTC_z) + 's'
                # cv2.putText(img, str(TTC)+'s', (int(c1_prex), int(c2_prey)), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
                break
            if v_x != 0:
                TTC = round(deltdistance_x / v_x, 2)
                # v_xTTC = str(v_x) + ' ' + str(TTC_x) + 's'
                # cv2.putText(img, str(TTC)+'s', (int(c1_prex), int(c2_prey)), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
                break
    result = [v_z, v_x]
    return result


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')

    parser.add_argument("--video", dest='video', help="Video to run detection upon",
                        default="200w4mm.mp4", type=str)
    parser.add_argument("--dataset", dest="dataset", help="Dataset on which the network has been trained",
                        default="pascal")
    parser.add_argument("--confidence", dest="confidence", help="Object Confidence to filter predictions", default=0.5)
    parser.add_argument("--nms_thresh", dest="nms_thresh", help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', help="weightsfile",
                        default="weights/yolov3-608.weights", type=str)
    parser.add_argument("--reso", dest='reso', help=
    "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default="416", type=str)
                        # default="608", type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 80
    bbox_attrs = 5 + num_classes

    print("Loading network.....")
    model = Darknet(args.cfgfile)
    model.load_weights(args.weightsfile)
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # torch.cuda.set_device(1) 只能在device 0
    if CUDA:
        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     model = nn.DataParallel(model, [0, 1])
        model.cuda()

    model.eval()  # 表示在用网络模型测试数据

    videofile = args.video

    videoReader = cv2.VideoCapture(videofile)
    fps = 0.0
    seconds = 0.0
    seconds1 = 0.0
    frames = 0
    flag = 0
    minfps = 100
    maxfps = 0
    count_less_than_25 = 0
    count_more_than_25 = 0

    resultsqueue = queue.Queue(60)
    secondsqueue = queue.Queue(60)

    start = time.time()
    while videoReader.isOpened():
        ret, frame = videoReader.read()
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 如果用imageio就需要转换一下，否则显示有问题
        start = time.time()
        img, orig_im, dim = prep_image(frame, inp_dim)

        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), CUDA)
        output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)

        if type(output) == int:
            # frames += 1
            # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            end = time.time()
            seconds = end - start
            fps = (fps + (1 / seconds)) / 2
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            continue

        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        target_num = output.shape[0]

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        classes = load_classes('data/voc.names')
        colors = pkl.load(open("pallete", "rb"))

        result_distance = list(map(lambda x: write(x, orig_im, classes, colors)[1], output))
        end = time.time()
        seconds = end - start

        if len(result_distance) > 0:
            flag = flag + 1
        if flag < 2:
            resultsqueue.put(result_distance)
            secondsqueue.put(seconds)
            seconds1 = seconds1 + seconds
        else:
            result_past = resultsqueue.get()
            secondsget = secondsqueue.get()
            resultsqueue.put(result_distance)
            secondsqueue.put(seconds)
            seconds1 = seconds1 + seconds - secondsget
            result_v = list(map(lambda x: cal_v(x, result_past, seconds1, orig_im), result_distance))
            print('2', end=',')  # 'type'
            print('0', end=',')  # service id
            print(round(end, 5), end=',')  # 'time:',
            print(target_num, end='')  # 'num:',
            for i in range(output.shape[0]):
                print(',', i, end=',')  # 'id:',
                print(result_distance[i][6], end=',')  # 'label:',
                print(result_distance[i][0], end=',')  # 'distance_z:',
                print(result_distance[i][1], end=',')  # 'distance_x:',
                print(result_v[i][0], end=',')  # 'speed_z:',
                print(result_v[i][1], end='')  # 'speed_x:',
        print()
        fps = (fps + (1 / seconds)) / 2
        print("FPS of the video is {:5.2f}".format(fps))
        # count fps message
        if minfps > fps:
            minfps = fps
        if maxfps < fps:
            maxfps = fps
        if 25 > fps:
            count_less_than_25 += 1
        if 25 <= fps:
            count_more_than_25 += 1
        print("Max FPS is {:5.2f}".format(maxfps), "count_more_than_25 is", count_more_than_25)
        print("Min FPS is {:5.2f}".format(minfps), "count_less_than_25 is", count_less_than_25)

        cv2.imshow("frame", orig_im)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        # frames += 1
        # print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
