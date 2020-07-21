
from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
from bbox import bbox_iou, bbox_iou1, bbox_iou_half

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def convert2cpu(matrix):
    if matrix.is_cuda:
        return torch.FloatTensor(matrix.size()).copy_(matrix)
    else:
        return matrix

def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    # anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)  #prediction  cuda
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    #Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    #Add the center offsets
    grid_len = np.arange(grid_size)
    a,b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1).cuda()
    y_offset = torch.FloatTensor(b).view(-1,1).cuda()
    
    # if CUDA:
    #     x_offset = x_offset.cuda()
    #     y_offset = y_offset.cuda()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset     # cx cy offset
      
    #log space transform height and the width
    anchors = torch.FloatTensor(anchors).cuda()
    
    # if CUDA:
    #     anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors     # w h

    #Softmax the class scores
    # prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5 : 5 + num_classes]))
    prediction[:, :, 5:bbox_attrs] = torch.sigmoid(prediction[:, :, 5:bbox_attrs])

    prediction[:,:,:4] *= stride     # cx cy w h
    
    return prediction

def predict_transform1(prediction, stride, anchors, num_anchors, bbox_attrs):
    batch_size = prediction.size(0)
    # stride = inp_dim // prediction.size(2)
    # grid_size = inp_dim // stride
    grid_size = prediction.size(2)
    # bbox_attrs = 5 + num_classes
    # num_anchors = len(anchors)

    # anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    prediction = prediction.view(batch_size, bbox_attrs * num_anchors, grid_size * grid_size)  # prediction  cuda
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)

    x_offset = torch.FloatTensor(a).view(-1, 1).cuda()
    y_offset = torch.FloatTensor(b).view(-1, 1).cuda()

    # if CUDA:
    #     x_offset = x_offset.cuda()
    #     y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, :2] += x_y_offset  # cx cy offset

    # log space transform height and the width
    anchors = torch.FloatTensor(anchors).cuda()   # 32-bit floating point

    # if CUDA:
    #     anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors  # w h

    # Softmax the class scores
    # prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5 : 5 + num_classes]))
    prediction[:, :, 5:bbox_attrs] = torch.sigmoid(prediction[:, :, 5:bbox_attrs])

    prediction[:, :, :4] *= stride  # cx cy w h

    return prediction

def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def get_im_dim(im):
    im = cv2.imread(im)
    w,h = im.shape[1], im.shape[0]
    return w,h

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def dynamic_write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    prediction_bak = prediction.clone()
    # dets = write_results(prediction.clone(), confidence, num_classes, nms, nms_conf)
    # dets = write_results_half(prediction.clone(), confidence, num_classes, nms, nms_conf)
    dets = write_results2(prediction.clone(), confidence, num_classes, nms, nms_conf)
    if isinstance(dets, int):
        return dets

    if dets.shape[0] > 100:
        nms_conf -= 0.05
        dets = write_results(prediction_bak.clone(), confidence, num_classes, nms, nms_conf)

    return dets

def write_results(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().float().unsqueeze(2)   # 1, 10674 ,1
    prediction = prediction * conf_mask   # 1,10647,85

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()  #2,n n为满足条件的个数
    except:
        return 0
    # xmin ymin xmax ymax
    box_a = prediction.new(prediction.shape)
    box_a[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2]/2)
    box_a[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3]/2)
    box_a[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2]/2)
    box_a[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3]/2)
    prediction[:, :, :4] = box_a[:, :, :4]

    batch_size = prediction.size(0)   # batch no.

    output = prediction.new(1, prediction.size(2) + 1)  # 1,86   5 + 80 + 1
    write = False
    num = 0
    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]   # 10674, 85

        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1) #10647
        max_conf = max_conf.float().unsqueeze(1)  # 10647,1
        max_conf_score = max_conf_score.float().unsqueeze(1) #10674, 1
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)   # 10674, 7  (xmin, xmax,ymin,ymax, conf, class_max_conf, class_max_conf_idx)

        # Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:, 4]))  # n, 1  conf不为0的行坐标, n为目标个数

        image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7) # 筛选出满足条件的目标  n,7

        # Get the various classes detected in the image
        try:
            img_classes = unique(image_pred_[:, -1])  # class idx 检测到的类别个数列表
        except:
            continue

        # WE will do NMS classwise
        # print(img_classes)
        for cls in img_classes:
            # if cls != 0: #0 is the person
            #     continue
            # get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:, -1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # sort the detections such that the entry with the maximum objectness
            # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)

            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            # if nms has to be done
            # if nms:
            #     # Perform non-maximum suppression
            #     max_detections = []
            #     while image_pred_class.size(0):
            #         # Get detection with highest confidence and save as max detection
            #         max_detections.append(image_pred_class[0].unsqueeze(0))
            #         # Stop if we're at the last detection
            #         if len(image_pred_class) == 1:
            #             break
            #         # Get the IOUs for all boxes with lower confidence
            #         ious = bbox_iou(max_detections[-1], image_pred_class[1:])
            #         # Remove detections with IoU >= NMS threshold
            #         image_pred_class = image_pred_class[1:][ious < nms_conf]
            #     image_pred_class = torch.cat(max_detections).data


            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output,out))
            num += 1
    
    if not num:
        return 0

    return output

def write_results1(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    prediction = prediction.squeeze()   # 10674, 85  -->  cx cy w h conf class_conf
    conf_mask = (prediction[:, 4] > confidence) # 10674
    image_pred = prediction[conf_mask].contiguous()   # n, 85  满足阈值的预测目标

    if len(image_pred) == 0:
        return 0

    #cx, cy, w, h  -->  xmin ymin xmax ymax
    box_a = image_pred.new(image_pred.shape)
    box_a[:, 0] = (image_pred[:, 0] - image_pred[:, 2] / 2)
    box_a[:, 1] = (image_pred[:, 1] - image_pred[:, 3] / 2)
    box_a[:, 2] = (image_pred[:, 0] + image_pred[:, 2] / 2)
    box_a[:, 3] = (image_pred[:, 1] + image_pred[:, 3] / 2)
    image_pred[:, :4] = box_a[:, :4]

    # output = image_pred.new(1, image_pred.size(1) + 1)  # 1,86   5 + 80 + 1

    write = False
    # num = 0

    max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], axis = 1)  # n
    max_conf = max_conf.float().unsqueeze(1)  # n,1
    max_conf_score = max_conf_score.float().unsqueeze(1)  # n, 1
    seq = (image_pred[:, :5], max_conf, max_conf_score)
    image_pred = torch.cat(seq, 1)  # n, 7  (xmin, xmax,ymin,ymax, conf, class_max_conf, class_max_conf_idx)
    image_pred_ = image_pred.view(-1, 7)
    img_classes = unique(image_pred_[:, -1])

    keep_boxes = []
    for cls in img_classes:
        # get the detections with one particular class
        cls_mask = (image_pred_[:, -1] == cls)
        image_pred_class = image_pred_[cls_mask].view(-1, 7)

        # if len(image_pred_class) == 1:
        #     continue
        # sort the detections such that the entry with the maximum objectness
        # max confidence is at the top
        conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
        image_pred_class = image_pred_class[conf_sort_index]
        # idx = image_pred_class.size(0)

        while len(image_pred_class) > 1:
            box, image_pred_class = image_pred_class[0], image_pred_class[1:]
            keep_boxes.append(box)
            ious = bbox_iou1(box, image_pred_class)
            # keep_boxes.append(image_pred_class[0])
            # ious = bbox_iou1(image_pred_class[0], image_pred_class[1:])
            iou_mask = (ious < nms_conf)
            # image_pred_class = image_pred_class[1:]
            image_pred_class = image_pred_class[iou_mask]
        else:
            if len(image_pred_class) != 0:
                keep_boxes.append(image_pred_class[0])

        # if not write:
        #     output = torch.stack(keep_boxes)
        #     write = True
        # else:
        #     out = torch.stack(keep_boxes)
        #     output = torch.cat((output, out))

        # batch_ind = keep_boxes.new(keep_boxes.size(0), 1).fill_(1)
        # seq = batch_ind, keep_boxes
        # if not write:
        #     output = torch.cat(seq, 1)
        #     write = True
        # else:
        #     out = torch.cat(seq, 1)
        #     output = torch.cat((output, out))

    output = torch.stack(keep_boxes)
    batch_ind = output.new(output.size(0), 1).fill_(0)
    output = torch.cat((batch_ind, output), 1)
    return output

def write_results2(prediction, confidence, num_classes, nms=True, nms_conf=0.4):
    prediction = prediction.squeeze()   # 10674, 85  -->  cx cy w h conf class_conf
    conf_mask = (prediction[:, 4] > confidence) # 10674
    image_pred = prediction[conf_mask].contiguous()   # n, 85  满足阈值的预测目标

    if len(image_pred) == 0:
        return 0

    #cx, cy, w, h  -->  xmin ymin xmax ymax
    box_a = image_pred.new(image_pred.shape)
    # print(image_pred.shape)
    box_a[:, 0] = (image_pred[:, 0] - image_pred[:, 2] / 2)
    box_a[:, 1] = (image_pred[:, 1] - image_pred[:, 3] / 2)
    box_a[:, 2] = (image_pred[:, 0] + image_pred[:, 2] / 2)
    box_a[:, 3] = (image_pred[:, 1] + image_pred[:, 3] / 2)
    image_pred[:, :4] = box_a[:, :4]

    # output = image_pred.new(1, image_pred.size(1) + 1)  # 1,86   5 + 80 + 1

    # write = False
    # num = 0

    max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)  # n
    max_conf = max_conf.float().unsqueeze(1)  # n,1
    max_conf_score = max_conf_score.float().unsqueeze(1)  # n, 1
    seq = (image_pred[:, :5], max_conf, max_conf_score)
    image_pred = torch.cat(seq, 1)  # n, 7  (xmin, xmax,ymin,ymax, conf, class_max_conf, class_max_conf_idx)
    image_pred_ = image_pred.view(-1, 7)
    # img_classes = unique(image_pred_[:, -1])

    keep_boxes = []
    conf_sort_index = torch.sort(image_pred_[:, 4], descending=True)[1]
    image_pred_sorted = image_pred_[conf_sort_index]   #按置信度进行排序
    while len(image_pred_sorted) > 1:
        box, image_pred_sorted = image_pred_sorted[0], image_pred_sorted[1:]
        keep_boxes.append(box)
        ious = bbox_iou1(box, image_pred_sorted)
        # keep_boxes.append(image_pred_class[0])
        # ious = bbox_iou1(image_pred_class[0], image_pred_class[1:])
        iou_mask = (ious < nms_conf)
        # image_pred_class = image_pred_class[1:]
        image_pred_sorted = image_pred_sorted[iou_mask]
    else:
        if len(image_pred_sorted) != 0:
            keep_boxes.append(image_pred_sorted[0])

    #
    # for cls in img_classes:
    #     # get the detections with one particular class
    #     cls_mask = (image_pred_[:, -1] == cls)
    #     image_pred_class = image_pred_[cls_mask].view(-1, 7)
    #
    #     # if len(image_pred_class) == 1:
    #     #     continue
    #     # sort the detections such that the entry with the maximum objectness
    #     # max confidence is at the top
    #     conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
    #     image_pred_class = image_pred_class[conf_sort_index]
    #     # idx = image_pred_class.size(0)
    #
    #     while len(image_pred_class) > 1:
    #         box, image_pred_class = image_pred_class[0], image_pred_class[1:]
    #         keep_boxes.append(box)
    #         ious = bbox_iou1(box, image_pred_class)
    #         # keep_boxes.append(image_pred_class[0])
    #         # ious = bbox_iou1(image_pred_class[0], image_pred_class[1:])
    #         iou_mask = (ious < nms_conf)
    #         # image_pred_class = image_pred_class[1:]
    #         image_pred_class = image_pred_class[iou_mask]
    #     else:
    #         if len(image_pred_class) != 0:
    #             keep_boxes.append(image_pred_class[0])
    #
    #     # if not write:
    #     #     output = torch.stack(keep_boxes)
    #     #     write = True
    #     # else:
    #     #     out = torch.stack(keep_boxes)
    #     #     output = torch.cat((output, out))
    #
    #     # batch_ind = keep_boxes.new(keep_boxes.size(0), 1).fill_(1)
    #     # seq = batch_ind, keep_boxes
    #     # if not write:
    #     #     output = torch.cat(seq, 1)
    #     #     write = True
    #     # else:
    #     #     out = torch.cat(seq, 1)
    #     #     output = torch.cat((output, out))

    output = torch.stack(keep_boxes)
    batch_ind = output.new(output.size(0), 1).fill_(0)
    output = torch.cat((batch_ind, output), 1)
    return output

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 00:12:16 2018

@author: ayooshmac
"""
def predict_transform_half(prediction, inp_dim, anchors, num_classes, CUDA = True):
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)

    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    grid_size = inp_dim // stride

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    # Sigmoid the  centre_X, centre_Y. and object confidencce
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])

    # Add the center offsets
    grid_len = np.arange(grid_size)
    a, b = np.meshgrid(grid_len, grid_len)
    
    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)
    
    # if CUDA:
    x_offset = x_offset.cuda().half()
    y_offset = y_offset.cuda().half()
    
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    prediction[:,:,:2] += x_y_offset
      
    # log space transform height and the width
    anchors = torch.HalfTensor(anchors)
    
    # if CUDA:
    anchors = anchors.cuda()
    
    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors

    # Softmax the class scores
    # prediction[:,:,5: 5 + num_classes] = nn.Softmax(-1)(Variable(prediction[:,:, 5 : 5 + num_classes])).data
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5: 5 + num_classes]))
    prediction[:,:,:4] *= stride

    return prediction


def write_results_half(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).half().unsqueeze(2)
    prediction = prediction*conf_mask
    prediction = prediction.half()

    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0

    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]

    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False
    
    for ind in range(batch_size):
        # select the image from the batch
        image_pred = prediction[ind]
        
        # Get the class having maximum score, and the index of that class
        # Get rid of num_classes softmax scores
        # Add the class index and the class score of class having maximum score
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.half().unsqueeze(1)
        max_conf_score = max_conf_score.half().unsqueeze(1)
        # pred = image_pred[:,:5].half()
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        
        # Get rid of the zero entries
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:]
        except:
            continue
        
        # Get the various classes detected in the image
        img_classes = unique(image_pred_[:,-1].long()).half()
        
        
        
                
        # WE will do NMS classwise
        for cls in img_classes:
            # get the detections with one particular class
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).half().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()

            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)

            # image_pred_class = image_pred_[class_mask_ind]

             # sort the detections such that the entry with the maximum objectness
             # confidence is at the top
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True )[1]

            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            # if nms has to be done
            if nms:
                # For each detection
                for i in range(idx):
                    # Get the IOUs of all boxes that come after the one we are looking at
                    # in the loop
                    try:
                        ious = bbox_iou_half(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])

                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    # Zero out all the detections that have IoU > treshhold
                    iou_mask = (ious < nms_conf).half().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    # Remove the non-zero entries
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)

            # Concatenate the batch_id of the image to the detection
            # this helps us identify which image does the detection correspond to
            # We use a linear straucture to hold ALL the detections from the batch
            # the batch_dim is flattened
            # batch is identified by extra batch column
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))
    
    return output
