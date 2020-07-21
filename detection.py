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
from data_sender import udp_sender_protobuf
from perceptron_pb2 import PerceptronSet
from bbox import bbox_iou


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
    #if cls == 0:
    #    real_class = 3
   # elif cls == 1 or cls == 3 or cls == 15 or cls == 16 or cls == 17 or cls ==18 or cls==19:
   #     real_class = 2
   # elif cls == 2 or cls == 5 or cls == 7:
   #     real_class = 1
    # new 类别
    if cls == 3:
        real_class = 3
    elif cls == 0 or cls == 1 or cls == 2:
        real_class = 2
    elif cls == 4 or cls == 5:
        real_class = 1

    x = x.cpu().numpy()
    result = [real_class, down_mid_pt, (v, v_angle), x[1:5], cls, conf]
    return img, result

def video_demo(frame, inp_dim, quadrangle, onnx2trt, deepsort, classes, colors, h_inv):

    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1, 2)
    im_dim = im_dim.cuda()


    start = time.time()
    output = onnx2trt.detect_thread(frame, img)

    end = time.time() - start

    if type(output) == int:
        return orig_im, []

    #rescale bbox  416,416 --> 1920 1080
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)  #
    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2
    output[:, 1:5] /= scaling_factor
    # target_num = output.shape[0]

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    # 针对性改进__BaiYu write_select 替代 write ，只显示四边形内部检测信息
    result_list = list(map(lambda x: write_select(x, orig_im, classes, colors, quadrangle)[1], output))


    # 所有类型的目标都跟踪
    bbox_Tracking = []      #矩形框
    cls_ids_Tracking = []   #类别下标
    cls_conf = []           #置信度

    for bi in range(len(result_list) - 1, -1, -1):
        # if result_list[bi][0] == 2 or result_list[bi][0] == 3:
        if result_list[bi][-1] <= 0:  #根据置信度，删掉不在ROI区域的目标
            continue
        bbox_Tracking.append(result_list[bi][3])
        cls_ids_Tracking.append(result_list[bi][4])
        cls_conf.append(result_list[bi][5])

    outputs_tracking = []
    # # if bbox_Tracking is not None:
    bbox_xcycwh = []

    # 转化为centerX, centerY, width, height bbox形式
    for i in range(len(bbox_Tracking)):
        (cx, cy) = ((bbox_Tracking[i][0] + bbox_Tracking[i][2]) / 2.0, (bbox_Tracking[i][1] + bbox_Tracking[i][3]) / 2.0)
        (w, h) = (bbox_Tracking[i][2] - bbox_Tracking[i][0], bbox_Tracking[i][3] - bbox_Tracking[i][1])
        bbox_xcycwh.append([cx, cy, w, h])

    bbox_xcycwh = np.asarray(bbox_xcycwh)
    cls_conf = np.asarray(cls_conf)
    # global deepsort
    if bbox_xcycwh is not None and len(bbox_xcycwh) > 0:
        outputs_tracking = deepsort.update(bbox_xcycwh, cls_conf, cls_ids_Tracking, frame)

    end = time.time()
    print('runtime: {0:.2f} ms'.format((end - start)*1000))
    
    if outputs_tracking is not None and len(outputs_tracking) > 0:
        # if len(boxes) > 0:
        bbox_xyxy = outputs_tracking[:, :4]   #x1, y1, x2, y2
        identities = outputs_tracking[:, 5]  #track_id
        clsTracking = outputs_tracking[:, 4]  #classLabel index
        trace = outputs_tracking[:, -1]   # trace of object
        #打印追踪后的框bbox  ids
        ori_im = draw_bboxes(frame, bbox_xyxy, identities, clsTracking, trace, h_inv)

    return orig_im, outputs_tracking
