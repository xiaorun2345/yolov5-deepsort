import argparse

from utils.datasets import *
from utils.utils import *

import numpy.linalg as lg
import pickle as pkl
import math
# from video_string_test_tensorrt import prep_image
from preprocess import prep_image
import numpy as np
import cv2
from util2 import load_classes
from deep_sort.util import draw_bboxes
import torch
import time
from parse_data import dict_protobuf
from data_sender import *
from perceptron_pb2 import PerceptronSet
from parse_data import parse_conf, parse_argument
from bbox import bbox_iou
from deep_sort import DeepSort
from datetime import datetime
from lidar_camera_utils import pnp_object_location
deepsort = DeepSort('./deep_sort/deep/checkpoint/ckpt.t7')
#########################################################################
def warp_perspective_point(pt_in, M):
    """
    :param pt_in:  是下中点 ((x1+x2)/2,y2)
    :param M:
    :return:
    """
    pt_in_x = pt_in[0]
    pt_in_y = pt_in[1]
    pt_out_x = (M[0][0] * pt_in_x + M[0][1] * pt_in_y + M[0][2]) / (M[2][0] * pt_in_x + M[2][1] * pt_in_y + M[2][2])
    pt_out_y = (M[1][0] * pt_in_x + M[1][1] * pt_in_y + M[1][2]) / (M[2][0] * pt_in_x + M[2][1] * pt_in_y + M[2][2])
    return pt_out_x, pt_out_y

# 重写透视变换代码，保证精度能够到float64
def get_perspective(org_points, get_points):
    """
    :param org_points: n * 2点集坐标(x, y)
    :param get_points: n * 2点集坐标(x, y), org_points
    :return: perspective_mat：3 * 3 get_points = perspective_mat * org_points
    """
    assert (len(org_points) == len(get_points)) and len(org_points) >= 4
    nums = len(org_points)
    coefficient_mat = np.zeros((2 * nums, 8))
    b = np.zeros(2 * nums)
    for i in range(nums):
        org_point = org_points[i]
        get_point = get_points[i]
        coefficient_mat[2 * i] = [org_point[0], org_point[1], 1,
                                  0, 0, 0,
                                  -org_point[0] * get_point[0], -get_point[0] * org_point[1]]
        b[2 * i] = get_point[0]

        coefficient_mat[2 * i + 1] = [0, 0, 0,
                                      org_point[0], org_point[1], 1,
                                      -org_point[0] * get_point[1], -org_point[1] * get_point[1]]
        b[2 * i + 1] = get_point[1]
    if np.linalg.det(coefficient_mat) == 0:
        return coefficient_mat
    coefficient_mat_inv = lg.inv(coefficient_mat)
    perspective_mat = np.dot(coefficient_mat_inv, b)  # 大于4个点时候为最小二乘法计算
    perspective_mat = np.append(perspective_mat, np.array([1]))
    perspective_mat = np.reshape(perspective_mat, (3, 3))
    return perspective_mat

def determinant(v1, v2, v3, v4):  # 行列式
    return v1 * v3 - v2 * v4

# 判断两条线断是否交叉
def intersect(aa, bb, cc, dd):
    delta = determinant(bb[0] - aa[0], dd[0] - cc[0], dd[1] - cc[1], bb[1] - aa[1])
    if -1e-6 <= delta <= 1e-6:  # delta=0，表示两线段重合或平行
        if aa[1] - cc[1] == 0 and aa[1] - dd[1] == 0:
            return True
        elif (aa[1] - cc[1]) * (aa[1] - dd[1]) == 0:
            return False
        else:
            if (aa[0] - cc[0]) / (aa[1] - cc[1]) == (aa[0] - dd[0]) / (aa[1] - dd[1]):
                return True
            return False
    lamuda = determinant(dd[0] - cc[0], aa[0] - cc[0], aa[1] - cc[1], dd[1] - cc[1]) / delta
    if lamuda > 1 or lamuda < 0:
        return False
    miu = determinant(bb[0] - aa[0], aa[0] - cc[0], aa[1] - cc[1], bb[1] - aa[1]) / delta
    if miu > 1 or miu < 0:
        return False
    return True

# 判断四边形是否有边交叉
def is_cross(pls):
    if intersect(pls[0], pls[3], pls[1], pls[2]):
        return 1
    if intersect(pls[0], pls[1], pls[2], pls[3]):
        return 2
    return 0

def cross_area(pt1, pt2, pt3, pt4=None):
    # 三角形面积计算公式
    if pt4 is None:
        ax = pt2[0] - pt1[0]
        ay = pt2[1] - pt1[1]
        bx = pt2[0] - pt3[0]
        by = pt2[1] - pt3[1]
        return math.fabs(ax * by - ay * bx) / 2
    # 凸四边形面积计算公式
    else:
        S1 = cross_area(pt1, pt2, pt3)
        S2 = cross_area(pt1, pt3, pt4)
        return S1 + S2

# 判断点是否在多边形内，如果在，则返回True,否则False
def is_pt_in_quadrangle(pt, pt1, pt2, pt3, pt4):
    S = cross_area(pt1, pt2, pt3, pt4)
    S1 = cross_area(pt1, pt2, pt)
    S2 = cross_area(pt2, pt3, pt)
    S3 = cross_area(pt3, pt4, pt)
    S4 = cross_area(pt4, pt1, pt)
    return math.fabs(S - S1 - S2 - S3 - S4) < 1

def drawM(img, pls):
    pts = pls.reshape((-1, 1, 2))
    # cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    cv2.polylines(img, [pts], True, (0, 0, 255), 3)

    return img

def draw_pt(img, pls):
    for i in pls:
        ti = tuple(i)
        if i is not (0, 0):
            # img_r = cv2.circle(img, ti, 3, (0, 255, 255), 3)
            img_r = cv2.circle(img, ti, 3, (0, 255, 0), 3)

    return img_r

# 已知图像，分类结果，分类个数，颜色选择范围，四边形范围，绘制四边形范围内的物体
def write_select(x, img, classes, colors, quadrangle):
    result = [0, (-1, -1), (0, 0), (0, 0, 0, 0), 0, 0] #类别，下边框中心点，速度和角度，矩形框，类别下标，置信度
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    conf = round(float(x[5]), 2)  #保留两位小数

    if c2[0] <= 0 or c2[1] <= 0 or c2[0]<=c1[0] or c2[1]<=c1[1]:
        return img, result

    #筛选目标类别，只保留2 car, 5 bus , 7 truck, 3 motorbike, 1 bicycle, 0 person
    #使用原始模型Coco数据集上，80个目标类别
    # coco : 0 person 1 bicycle  2 car 3 motorbike 5 bus 7 truck
    # if not (cls == 2 or cls == 5 or cls == 7 or cls == 3 or cls == 1 or cls == 0):
    #     return img, result

    # 计算矩形框的下中点，如果此点在四边形内，则画矩形，否则跳出
    down_mid_pt = (int((c1[0] + c2[0]) / 2), int(c2[1]))
    ul_pt = (quadrangle[0][0], quadrangle[0][1])
    ur_pt = (quadrangle[1][0], quadrangle[1][1])
    dr_pt = (quadrangle[2][0], quadrangle[2][1])
    dl_pt = (quadrangle[3][0], quadrangle[3][1])
    if not is_pt_in_quadrangle(down_mid_pt, ul_pt, ur_pt, dr_pt, dl_pt):
        return img, result

    # find out label of target
    label = "{0}:{1:.2f}".format(classes[cls], conf)
    # draw object rect
    # color = random.choice(colors)
    color = colors[cls]
    cv2.rectangle(img, c1, c2, color, 2)
    # draw text rect
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.1, 2)[0]
    c2_new = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2_new, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX, 1.1, [225, 255, 255], 2)

    v = 0
    v_angle = 0

    real_class = 0
    # coco 类别
    if cls == 0:
        real_class = 3
    elif cls == 1 or cls == 3 or cls == 15 or cls == 16 or cls == 17 or cls ==18 or cls==19:
        real_class = 2
    elif cls == 2 or cls == 5 or cls == 7:
        real_class = 1
    # new 类别
    # if cls == 3:
    #     real_class = 3
    # elif cls == 0 or cls == 1 or cls == 2:
    #     real_class = 2
    # elif cls == 4 or cls == 5:
    #     real_class = 1

    x = x.cpu().numpy()
    result = [real_class, down_mid_pt, (v, v_angle), x[1:5], cls, conf]
    return img, result

#########################################################################



def detect(Sensor_ID, udpClient, addr,jsonName):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam =source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    solve_homegraphy = pnp_object_location(jsonName)
    object_2d_points, object_3d_point = solve_homegraphy.com_cfg()
    h, h_inv = solve_homegraphy.solve_Hom(object_2d_points, object_3d_point)
    # Initialize
    device = torch_utils.select_device(opt.device)
    half =True  # half precision only supported on CUDA
   
    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)['model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    t0 = time.time()
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
    # 所有类型的目标都跟踪

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')

    out__ = cv2.VideoWriter('./output.avi',fourcc,20,(1280,720))
    # Run inference
    
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half())
    for path, img, im0s, vid_cap in dataset:
        bbox_Tracking = []      #矩形框
        cls_ids_Tracking = []   #类别下标
        cls_conf = []           #置信度
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        outputs_tracking = []
        # Inference
        t1 = time.time()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy=np.asarray((torch.tensor(xyxy).view(1, 4))).astype(int)[0]
                    cxcy=list([int((xyxy[0]+xyxy[2])/2),int((xyxy[1]+xyxy[3])/2),int(xyxy[2]-xyxy[0]),int(xyxy[3]-xyxy[1])])
                    bbox_Tracking.append(cxcy)
                    cls_ids_Tracking.append(cls)
                    cls_conf.append(conf)
                    #label = '%s %.2f' % (names[int(cls)], conf)
                    #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
            bbox_Tracking=np.asarray(bbox_Tracking)
            cls_ids_Tracking=np.asarray(cls_ids_Tracking)
            cls_conf=np.asarray(cls_conf)
            global deepsort
            if len(bbox_Tracking) > 0:
                outputs_tracking = deepsort.update(bbox_Tracking, cls_conf, cls_ids_Tracking, im0)
            if outputs_tracking is not None and len(outputs_tracking) > 0:
            # if len(boxes) > 0:
                bbox_xyxy = outputs_tracking[:, :4]   #x1, y1, x2, y2
                identities = outputs_tracking[:, 5]  #track_id
                clsTracking = outputs_tracking[:, 4]  #classLabel index
                trace = outputs_tracking[:, -1]   # trace of object
                #打印追踪后的框bbox  ids
                ori_im = draw_bboxes(im0, bbox_xyxy, identities, clsTracking, trace, h_inv)
                
                resultSender(outputs_tracking, h_inv, Sensor_ID, udpClient, addr, (time.time()-t1))
            t2=time.time()
                # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if True:
                cv2.imshow(p, im0)
                print(im0.shape)
                out__.write(im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights-s/best.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/videos/getvideo_189_2019-07-20_09-17-48.avi', help='source')  # file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='rtsp://admin:hik12345@192.168.1.64:554//Streaming/Channels/101', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--idx', default='2', help='idx')
    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        
        Sensor_ID, local_IP, local_port, heart_port, cameraAddr, cam_IP, pls_xy, real_xy, RSU_IP, RSU_port, jsonName, is_show = parse_conf(
            'NL_config', opt.idx)
        udpClient, addr = udp_client(RSU_IP, RSU_port)
        detect(Sensor_ID, udpClient, addr,jsonName)
