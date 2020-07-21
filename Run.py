# -*- coding: utf-8 -*-
from video_real import Ui_Form
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QTabWidget
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtGui import QPixmap, QImage
# from PyQt5.QtWidgets import QLabel, QWidget
from video_string_test import prep_image, arg_parse, cal_v
import sys, time, os
import cv2
# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import numpy as np
import numpy.linalg as lg
from util import *
from darknet import Darknet
# import pandas as pd
import pickle as pkl
# import random
# import argparse
# import imageio
import math
# import string
import queue
import configparser
import socket
import struct

global le_ul_x, le_ul_y, le_ur_x, le_ur_y, le_dl_x, le_dl_y, le_dr_x, le_dr_y
global p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y
global local_IP, local_port, RSU_IP, RSU_port, cam_IP, cam_port
local_IP = local_port = RSU_IP = RSU_port = cam_IP = cam_port = ''
le_ul_x = le_ul_y = le_ur_x = le_ur_y = le_dl_x = le_dl_y = le_dr_x = le_dr_y = '0'
p1_x = p1_y = p2_x = p2_y = p3_x = p3_y = p4_x = p4_y = 0
global Sensor_ID
Sensor_ID = ""
global Frame
global Ret
global Get_Flag
Frame = []
Ret = False
Get_Flag = True
global count_c
count_c = 0
conf = configparser.ConfigParser()

class TargetX:
    # 38个byte
    # example:t_x_x = [3, 1, 3, '0123456789012345', 45.3333354, 74.2352646, 'N', 'W', 34.223, 21.1234567]
    def __init__(self, t_x):
        # 2 byte|||short int/ int16 2 byte
        self.id = t_x[0]
        # 1 byte||| 0=unknown, 1=motor, 2=non_motor, 3=pedestrian
        self.type = t_x[1]
        # 1 byte||| 0=unknown，2=v2x，3=video，4=microwaveRadar，5=loop，6=merge
        self.sensor_type = t_x[2]
        # 16 byte|||某一类检测器的编号，如为融合后的目标，则此字段不填
        self.sensor_id = t_x[3]
        # 4 byte|||float 经度值
        self.longitude_value = t_x[4]
        # 4 byte|||float 纬度值
        self.latitude_value = t_x[5]
        # 1 byte|||char 纬度扩展 南纬北纬，N表示北纬，S表示南纬
        self.latitude_exp = t_x[6]
        # 1 byte|||char 经度扩展 东经西经，E表示东经，W表示西经
        self.longitude_exp = t_x[7]
        # 4 byte|||float 速度值
        self.v_value = t_x[8]
        # 4 byte|||float 速度方向 运动方向与正北方向的顺时针夹角 单位：度 0-360
        self.v_angle = t_x[9]

    def get_16byte_info(self):
        sensor_id_pieces = []
        for i in self.sensor_id:
            sensor_id_pieces.append(bytes(i, encoding='utf-8'))
        latitude_exp_b = bytes(self.latitude_exp, encoding='utf-8')
        longitude_exp_b = bytes(self.longitude_exp, encoding='utf-8')
        # print('sensor_id_pieces' + sensor_id_pieces)
        sensor_id_pieces_len = len(sensor_id_pieces)
        # print("sid len1 =", sensor_id_pieces_len)
        if sensor_id_pieces_len < 16:
            sensor_id_pieces.extend([b'0'] * (16 - sensor_id_pieces_len))
        # print(sensor_id_pieces)
        sensor_id_pieces_len = len(sensor_id_pieces)
        # print("sid len2 =", sensor_id_pieces_len)
        buf_all = struct.pack('!hBB16c2f2c2f', self.id, self.type, self.sensor_type,
                              sensor_id_pieces[0], sensor_id_pieces[1], sensor_id_pieces[2],
                              sensor_id_pieces[3], sensor_id_pieces[4], sensor_id_pieces[5],
                              sensor_id_pieces[6], sensor_id_pieces[7], sensor_id_pieces[8],
                              sensor_id_pieces[9], sensor_id_pieces[10], sensor_id_pieces[11],
                              sensor_id_pieces[12], sensor_id_pieces[13], sensor_id_pieces[14],
                              sensor_id_pieces[15], self.longitude_value, self.latitude_value,
                              latitude_exp_b, longitude_exp_b, self.v_value, self.v_angle)
        return buf_all


def warp_perspective_point(pt_in, M):
    pt_in_x = pt_in[0]
    pt_in_y = pt_in[1]
    pt_out_x = (M[0][0] * pt_in_x + M[0][1] * pt_in_y + M[0][2]) / (M[2][0] * pt_in_x + M[2][1] * pt_in_y + M[2][2])
    pt_out_y = (M[1][0] * pt_in_x + M[1][1] * pt_in_y + M[1][2]) / (M[2][0] * pt_in_x + M[2][1] * pt_in_y + M[2][2])
    return pt_out_x, pt_out_y


# 重写透视变换代码，保证精度能够到float64
def get_perspective(org_points, get_points):
    '''
    :param org_points: n * 2点集坐标(x, y)
    :param get_points: n * 2点集坐标(x, y), org_points
    :return: perspective_mat：3 * 3 get_points = perspective_mat * org_points
    '''

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
    # print(coefficient_mat)
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


# 已知图像，分类结果，分类个数，颜色选择范围，四边形范围，绘制四边形范围内的物体
def write_select(x, img, classes, colors, quadrangle):
    result = [0, (-1, -1), (0, 0), []]
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    conf = round(float(x[5]), 2)
    # 计算矩形框的下中点，如果此点在四边形内，则画矩形，否则跳出
    down_mid_pt = (int((c1[0] + c2[0]) / 2), int(c2[1]))
    ul_pt = (quadrangle[0][0], quadrangle[0][1])
    ur_pt = (quadrangle[1][0], quadrangle[1][1])
    dr_pt = (quadrangle[2][0], quadrangle[2][1])
    dl_pt = (quadrangle[3][0], quadrangle[3][1])

    if not is_pt_in_quadrangle(down_mid_pt, ul_pt, ur_pt, dr_pt, dl_pt):
        return result, ''
    # find out label of target
    cls = int(x[-1])
    label = "{0}:{1}".format(classes[cls],  conf)
    # draw object rect
    # color = random.choice(colors)
    color = colors[cls]
    cv2.rectangle(img, c1, c2, color, 2)
    # draw text rect
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
    c2_new = c1[0] + t_size[0] + 6, c1[1] + t_size[1] + 8
    cv2.rectangle(img, c1, c2_new, color, -1)
    # # caldulate distance z
    # width = img.shape[1]
    # height = img.shape[0]
    # pi = 3.1415926
    # f = 0.04  # focal length
    # d1 = 2.0  # the distance between the closest place the camera can shot and the camera
    # H = 1.5  # the height of the camera
    # scale = 0.00003  # Pixel size
    # beta1 = math.atan(d1 / H)
    # theta = math.atan(height * scale / (2 * f))
    # y2 = c2[1].item()
    # y = height - y2
    #
    # if y < 0.5 * height:
    #     alpha = theta - math.atan((0.5 * height * scale - y * scale) / f)
    # elif y > 0.5 * height:
    #     alpha = theta + math.atan((y * scale - 0.5 * height * scale) / f)
    # else:
    #     alpha = theta
    # if (c2[1] - c1[1]) != 0:
    #     b1 = (y * scale * math.sqrt(H * H + d1 * d1) / math.sqrt(f * f + (height * scale / 2) * (height * scale / 2)))
    #     d2 = (b1 * math.sin(pi / 2 - theta + alpha)) / math.sin(pi / 2 - alpha - beta1)
    #     distance_z = round(d1 + d2, 2)
    # else:
    #     distance_z = 0
    #
    # # calculate distance x
    # gama = 81.5  # horizontal angle of camera
    # W = 3  # 1/2 width of the closest place that the camera can shot
    # x1 = c1[0].item()
    # x2 = c2[0].item()
    # x_kernel = 1 / 2 * (x1 + x2)
    # distance_x = round(2 * pi * gama / 360 * distance_z / width * (x_kernel - 1 / 2 * width), 2)
    #
    # labdis = label + ' ' + str(distance_z) + ' ' + str(distance_x)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX, 1, [225, 255, 255], 2)
    # cv2.putText(img, labdis, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1)
    # result = [distance_z, distance_x, x[1].item(), x[2].item(), x[3].item(), x[4].item(), label]
    v = 0
    v_angle = 0
    # 0 = unknown，1 = motor，2 = non - motor，3 = pedestrian
    # voc datasets
    # non-motor type : bicycle  anaimal(cat cow dog horse sheep)
    # motor type : bus car motorbike train
    #    real_class = 0
    #    if cls == 14:
    #        real_class = 3
    #    elif cls == 1 or cls == 7 or cls == 9 or cls == 11 or cls == 12 or cls == 16:
    #        real_class = 2
    #    elif cls == 5 or cls == 6 or cls == 13 or cls == 18:
    #        real_class = 1

    # coco datasets
    # 2 non-motor:  bicycle motorbike
    # 1 motor：bus truck car
    # 0 unknown : cat dog horse sheep cow ...
    # 3 pedestrian
    # real_class = 0
    # if cls == 0:
    #     real_class = 3
    # elif cls == 1 or cls == 3:
    #     real_class = 2
    # elif cls == 2 or cls == 5 or cls == 7:
    #     real_class = 1

    #new data
    real_class = 0
    if cls == 3:
        real_class = 3
    elif cls == 4 or cls == 5:
        real_class = 2
    elif cls == 0 or cls == 1 or cls == 2:
        real_class = 1

#    else:
#        real_class = 0
    result = [real_class, down_mid_pt, (v, v_angle), [c1, c2, conf]]
    
    return result, label

#计算两个目标框的重合度
def calculate_overlap(boxA, boxB):
    x_start = max(boxA[0], boxB[0])
    y_start = max(boxA[1], boxB[1])
    x_end = min(boxA[2], boxB[2])
    y_end = min(boxA[3], boxB[3])

    interArea = max(0, x_end-x_start) * max(0, y_end-y_start)
    aArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    bArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # minArea = min(aArea, bArea)  # 可以排除大目标中的小目标，但会把真正的目标去掉
    # iou = interArea / minArea

    sumArea = aArea + bArea    # IOU
    iou = interArea / (sumArea - interArea)

    return iou

#只筛选出car, bus, truck, motorbike, bicycle, person目标
#删除车内检测出来的行人
def select_obj(img, output, classes, colors, quadrangle):
    bboxes = output[:, 1:5]
    cls_idx = output[:, -1]
    mask1 = (cls_idx == 0)  #person
    mask2 = (cls_idx == 2)  #car
    mask3 = (cls_idx == 5)  #bus
    mask4 = (cls_idx == 7)  #truck
    mask5 = (cls_idx == 1)  #bicycle
    mask6 = (cls_idx == 3)  #motorbike
    personList = bboxes[mask1]
    cls_p = cls_idx[mask1]
    persons = np.c_[personList, cls_p]
    mask = []
    for a, b, c in zip(mask2, mask3, mask4):
        if b:
            mask.append(b)
        elif c:
            mask.append(c)
        else:
            mask.append(a)
    motorList = bboxes[mask]
    cls_motor = cls_idx[mask]
    motors = np.c_[motorList, cls_motor]
    mask = []
    for a, b in zip(mask5, mask6):
        if b:
            mask.append(b)
        else:
            mask.append(a)
    non_motorList = bboxes[mask]
    cls_non = cls_idx[mask]
    non_motors = np.c_[non_motorList, cls_non]
    mask = []
    if len(personList) > 0 and len(motorList) > 0:
        for p in personList:
            flag = 0
            for m in motorList:
                iou = calculate_overlap(p, m)
                if iou > 0.7:
                    flag = 1
                    break
            if flag:
                mask.append(False)
            else:
                mask.append(True)
        persons = persons[mask]

    selectObj = []
    selectObj.extend(motors)
    selectObj.extend(persons)
    selectObj.extend(non_motors)

    result_list = []
    label_list = []
    for x in selectObj:
        box = x[0:4].astype(int)
        cls = int(x[-1])
        down_mid_pt = ((box[0] + box[2]) // 2, box[3])
        ul_pt = (quadrangle[0][0], quadrangle[0][1])
        ur_pt = (quadrangle[1][0], quadrangle[1][1])
        dr_pt = (quadrangle[2][0], quadrangle[2][1])
        dl_pt = (quadrangle[3][0], quadrangle[3][1])

        if is_pt_in_quadrangle(down_mid_pt, ul_pt, ur_pt, dr_pt, dl_pt):
            color = colors[cls]
            label = "{0}".format(classes[cls])
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.1, 2)[0]
            c2_new = box[0] + t_size[0] + 6, box[1] + t_size[1] + 18
            cv2.rectangle(img, (box[0], box[1]), c2_new, color, -1)
            cv2.putText(img, label, (box[0], box[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX, 1.1, [225, 255, 255], 2)
            v = 0
            v_angle = 0
            real_class = 0
            if cls == 0:
                real_class = 3
            elif cls == 1 or cls == 3:
                real_class = 2
            elif cls == 2 or cls == 5 or cls == 7:
                real_class = 1
            result_list.append([real_class, down_mid_pt, (v, v_angle)])
            label_list.append(label)

    return result_list, label_list

class mywindow(QTabWidget, Ui_Form):  # 这个窗口继承了用QtDesignner 绘制的窗口
    def __init__(self):
        super(mywindow, self).__init__()
        self.setupUi(self)
        self.label.setMouseTracking(True)  # 在label中开启鼠标跟踪
        self.th_v = None
        self.th_video = None
        self.cap_video = None
        self.th_c = None
        self.th_camera = None
        self.cap_camera = None
        self.ip_list = [[], [], []]

    def quit(self):
        button = QtWidgets.QMessageBox.question(self, "Question", "确认退出？",
                                                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                                                QtWidgets.QMessageBox.Ok)

        if button == QtWidgets.QMessageBox.Ok:
            if self.OpenVideo.text() == "CloseVideo":
                self.OpenVideo.setText("OpenVideo")
                self.th_video.stop()
                self.th_v.stop()
                self.th_video.change_pix_map.disconnect()
                self.th_video.change_pix_map2.disconnect()
                self.th_video.update_date.disconnect()
                self.th_video.udp_send.disconnect()
            if self.OpenCamera.text() == "CloseCamera":
                self.OpenCamera.setText("OpenCamera")
                self.th_camera.stop()
                self.th_c.stop()
                self.th_camera.change_pix_map.disconnect()
                self.th_camera.change_pix_map2.disconnect()
                self.th_camera.update_date.disconnect()
                self.th_camera.udp_send.disconnect()
                self.cap_camera.release()
            q_app = QtWidgets.QApplication.instance()
            q_app.quit()
        elif button == QtWidgets.QMessageBox.Cancel:
            pass

    def message_info(self, str_get):
        QtWidgets.QMessageBox.information(self, "Information", str_get)

    def mousePressEvent(self, mouse_event: QtGui.QMouseEvent):  # 鼠标点击事件
        # globalPos = self.mapToGlobal(mouse_event)  # 窗口坐标转换为屏幕坐标
        if mouse_event.button() == Qt.LeftButton:  # 按下鼠标左键
            if 0 <= mouse_event.x() <= 1120:
                self.Position_x.setText(str(mouse_event.x()))
            else:
                self.Position_x.setText("out of range")
            if 0 <= mouse_event.y() <= 630:
                self.Position_y.setText(str(mouse_event.y()))
            else:
                self.Position_y.setText("out of range")
        # do something
        # if a0.button() == Qt.RightButton:  # 按下鼠标右键
        # # do something
        # if a0.button() == Qt.MidButton:  # 按下鼠标中键
        # # do something

    def get_ip(self):
        # 获取本机ip
        self.lineEdit_local_ip.clear()
        udp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            udp_s.connect(('8.8.8.8', 80))
            my_addr = udp_s.getsockname()[0]
            self.lineEdit_local_ip.setText(str(my_addr))
        except Exception as ret:
            # 若无法连接互联网使用，会调用以下方法
            try:
                my_addr = socket.gethostbyname(socket.gethostname())
                self.lineEdit_local_ip.setText(str(my_addr))
            except Exception as ret_e:
                QtWidgets.QMessageBox.information(self, "Information", "无法获取ip，请连接网络！")
        finally:
            udp_s.close()

    def udp_sender_switch(self):
        if self.openudpButton.text() == "Open UDP":
            self.openudpButton.setText("Close UDP")
        else:
            self.openudpButton.setText("Open UDP")

    def udp_sender(self, data_list):
        if self.openudpButton.text() == "Close UDP":
            addr = (self.lineEdit_rsu_ip.text(), int(self.lineEdit_rsu_port.text()))
            udp_s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            data_s = ''
            data_head = b'\x0a\x0b\x0a\x0b'
            data_s += data_head.hex()
            target_number = len(data_list)
            target_number_b = struct.pack('!B', target_number)
            data_s += target_number_b.hex()
            for t_x_x_pieces in data_list:
                tx_pieces = TargetX(t_x_x_pieces)
                data_s += tx_pieces.get_16byte_info().hex()
            #print(data_s)
            data = bytes.fromhex(data_s)
            udp_s.sendto(data, addr)
            udp_s.close()
        else:
            pass

    def edit_conf(self, filename):
        #        conf.remove_section(filename)
        #        conf.add_section(filename)  # 添加section
        global local_IP, local_port, RSU_IP, RSU_port, cam_IP, cam_port
        global le_ul_x, le_ul_y, le_ur_x, le_ur_y, le_dl_x, le_dl_y, le_dr_x, le_dr_y
        global p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y
        global Sensor_ID
        Sensor_ID = self.SensorID.text()
        local_IP = self.lineEdit_local_ip.text()
        local_port = self.lineEdit_local_port.text()
        RSU_IP = self.lineEdit_rsu_ip.text()
        RSU_port = self.lineEdit_rsu_port.text()
        cam_IP = self.lineEdit_cam_ip.text()
        cam_port = self.lineEdit_cam_port.text()
        le_ul_x = self.LE_UL_x.text()
        le_ul_y = self.LE_UL_y.text()
        le_ur_x = self.LE_UR_x.text()
        le_ur_y = self.LE_UR_y.text()
        le_dl_x = self.LE_DL_x.text()
        le_dl_y = self.LE_DL_y.text()
        le_dr_x = self.LE_DR_x.text()
        le_dr_y = self.LE_DR_y.text()
        # 为配置文件添加值
        conf.set(filename, 'Sensor_ID', Sensor_ID)
        conf.set(filename, 'local_IP', local_IP)
        conf.set(filename, 'local_port', local_port)
        conf.set(filename, 'RSU_IP', RSU_IP)
        conf.set(filename, 'RSU_port', RSU_port)
        conf.set(filename, 'cam_IP', cam_IP)
        conf.set(filename, 'cam_port', cam_port)
        conf.set(filename, 'le_ul_x', le_ul_x)
        conf.set(filename, 'le_ul_y', le_ul_y)
        conf.set(filename, 'le_ur_x', le_ur_x)
        conf.set(filename, 'le_ur_y', le_ur_y)
        conf.set(filename, 'le_dl_x', le_dl_x)
        conf.set(filename, 'le_dl_y', le_dl_y)
        conf.set(filename, 'le_dr_x', le_dr_x)
        conf.set(filename, 'le_dr_y', le_dr_y)
        conf.set(filename, 'p1_x', str(p1_x))
        conf.set(filename, 'p1_y', str(p1_y))
        conf.set(filename, 'p2_x', str(p2_x))
        conf.set(filename, 'p2_y', str(p2_y))
        conf.set(filename, 'p3_x', str(p3_x))
        conf.set(filename, 'p3_y', str(p3_y))
        conf.set(filename, 'p4_x', str(p4_x))
        conf.set(filename, 'p4_y', str(p4_y))
        # 写入文件
        with open(filename + '.ini', 'w') as fw:
            conf.write(fw)

    def check_conf(self, filename):
        conf.read(filename + ".ini")
        global local_IP, local_port, RSU_IP, RSU_port, cam_IP, cam_port, cam_username, cam_password
        global le_ul_x, le_ul_y, le_ur_x, le_ur_y, le_dl_x, le_dl_y, le_dr_x, le_dr_y
        global p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y
        global Sensor_ID
        local_IP = conf.get(filename, "local_IP")
        local_port = conf.get(filename, 'local_port')
        RSU_IP = conf.get(filename, "RSU_IP")
        RSU_port = conf.get(filename, 'RSU_port')
        cam_IP = conf.get(filename, "cam_IP")
        cam_port = conf.get(filename, 'cam_port')
        cam_username = conf.get(filename, 'cam_username')
        cam_password = conf.get(filename, 'cam_password')
        self.SensorID.setText(conf.get(filename, "Sensor_ID"))
        self.lineEdit_local_ip.setText(conf.get(filename, "local_IP"))
        self.lineEdit_local_port.setText(conf.get(filename, 'local_port'))
        self.lineEdit_rsu_ip.setText(conf.get(filename, "RSU_IP"))
        self.lineEdit_rsu_port.setText(conf.get(filename, 'RSU_port'))
        self.lineEdit_cam_ip.setText(conf.get(filename, "cam_IP"))
        self.lineEdit_cam_port.setText(conf.get(filename, 'cam_port'))
        self.ip_list[0].append(conf.get(filename, "local_IP"))
        self.ip_list[0].append(conf.get(filename, "local_port"))
        self.ip_list[1].append(conf.get(filename, "RSU_IP"))
        self.ip_list[1].append(conf.get(filename, "RSU_port"))
        self.ip_list[2].append(conf.get(filename, "cam_IP"))
        self.ip_list[2].append(conf.get(filename, "cam_port"))
        le_ul_x = conf.get(filename, "le_ul_x")
        le_ul_y = conf.get(filename, "le_ul_y")
        le_ur_x = conf.get(filename, "le_ur_x")
        le_ur_y = conf.get(filename, "le_ur_y")
        le_dl_x = conf.get(filename, "le_dl_x")
        le_dl_y = conf.get(filename, "le_dl_y")
        le_dr_x = conf.get(filename, "le_dr_x")
        le_dr_y = conf.get(filename, "le_dr_y")
        self.LE_UL_x.setText(le_ul_x)
        self.LE_UL_y.setText(le_ul_y)
        self.LE_UR_x.setText(le_ur_x)
        self.LE_UR_y.setText(le_ur_y)
        self.LE_DL_x.setText(le_dl_x)
        self.LE_DL_y.setText(le_dl_y)
        self.LE_DR_x.setText(le_dr_x)
        self.LE_DR_y.setText(le_dr_y)
        p1_x = int(conf.get(filename, "p1_x"))
        p1_y = int(conf.get(filename, "p1_y"))
        p2_x = int(conf.get(filename, "p2_x"))
        p2_y = int(conf.get(filename, "p2_y"))
        p3_x = int(conf.get(filename, "p3_x"))
        p3_y = int(conf.get(filename, "p3_y"))
        p4_x = int(conf.get(filename, "p4_x"))
        p4_y = int(conf.get(filename, "p4_y"))

    def down_load_config(self):
        if not os.path.exists("NL_config.ini"):
            QtWidgets.QMessageBox.information(self, "Information", "NL_config.ini,请先创建")
        else:
            button = QtWidgets.QMessageBox.question(self, "Question", "确认覆盖现有配置？",
                                                    QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                                                    QtWidgets.QMessageBox.Ok)

            if button == QtWidgets.QMessageBox.Ok:
                self.check_conf('NL_config')
                QtWidgets.QMessageBox.information(self, "Information", "成功导入配置文件！")
            elif button == QtWidgets.QMessageBox.Cancel:
                pass

    def up_load_config(self):
        if not os.path.exists("NL_config.ini"):
            self.edit_conf('NL_config')
            QtWidgets.QMessageBox.information(self, "Information", "成功创建配置文件！")
        else:
            button = QtWidgets.QMessageBox.question(self, "Question", "确认覆盖原始配置文档？",
                                                    QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel,
                                                    QtWidgets.QMessageBox.Ok)

            if button == QtWidgets.QMessageBox.Ok:
                self.edit_conf('NL_config')
                QtWidgets.QMessageBox.information(self, "Information", "成功保存配置文件！")
            elif button == QtWidgets.QMessageBox.Cancel:
                pass

    def handle_display(self, data):
        self.NumofObject.setText(data)

    def video_processing(self):
        print(self.OpenVideo.text())
        if self.OpenVideo.text() == "OpenVideo":
            global videoName  # 在这里设置全局变量以便在线程中使用
            global Sensor_ID
            Sensor_ID = self.SensorID.text()
            # 返回路径下视频的全名称
            # videoName, videoType = QFileDialog.getOpenFileName(self,
            #                                                    "打开视频",
            #                                                    "./201805101500.mp4",
            #                                                    " *.mp4;;*.avi;;All Files (*)")
            videoName = '/home/user/yy/data/20190911-1828.avi'
            print("get video from", videoName)
            self.th_v = FrameReadLoop(videoName, "v")

            self.th_video = Thread()
            self.th_video.change_pix_map.connect(self.set_image)
            self.th_video.change_pix_map2.connect(self.set_image2)
            self.th_video.update_date.connect(self.handle_display)
            self.th_video.udp_send.connect(self.udp_sender)
            self.th_v.start()
            self.th_video.start()
            self.OpenVideo.setText("CloseVideo")
        else:
            self.OpenVideo.setText("OpenVideo")
            self.th_video.stop()
            self.th_v.stop()
            self.th_video.change_pix_map.disconnect()
            self.th_video.change_pix_map2.disconnect()
            self.th_video.update_date.disconnect()
            self.th_video.udp_send.disconnect()

    def camera_processing(self):
        print(self.OpenCamera.text())
        global CameraFlag
        CameraFlag = 1
        if self.OpenCamera.text() == "OpenCamera":
            CameraFlag = 1
            global Sensor_ID
            Sensor_ID = self.SensorID.text()
            camera_addr = "rtsp://" + cam_username + ":" + cam_password + "@" + self.lineEdit_cam_ip.text() + ":" + self.lineEdit_cam_port.text() + "//Streaming/Channels/1"
            #camera_addr = 0
            # camera_addr = '/home/user/yy/workspace/dataSet/jianzhou_data/3/out.avi'
            # camera_addr = '/home/user/yy/workspace/dataSet/jianzhou_data/1/original.avi'
            print("get Camera from", camera_addr)

            self.cap_camera = cv2.VideoCapture(camera_addr)
            fps = self.cap_camera.get(cv2.CAP_PROP_FPS)
            print("fps =", fps)
            flag = self.cap_camera.isOpened()
            if flag:
                self.th_c = FrameReadLoop(camera_addr, "c")  # 读视频线程
                self.cap_camera.release()
                self.th_camera = Thread()  # 播放视频线程
                self.th_camera.change_pix_map.connect(self.set_image)
                self.th_camera.change_pix_map2.connect(self.set_image2)
                self.th_camera.update_date.connect(self.handle_display)
                self.th_camera.udp_send.connect(self.udp_sender)
                self.th_c.warning_message.connect(self.message_info)
                self.th_c.start()
                self.th_camera.start()
                self.OpenCamera.setText("CloseCamera")
            else:
                QtWidgets.QMessageBox.information(self, "Warning", "请检测相机与电脑是否连接正确")
                self.OpenCamera.setText("OpenCamera")
                self.cap_camera.release()
        else:
            CameraFlag = 0
            self.OpenCamera.setText("OpenCamera")
            self.th_camera.stop()
            self.th_c.stop()
            self.th_camera.change_pix_map.disconnect()
            self.th_camera.change_pix_map2.disconnect()
            self.th_camera.update_date.disconnect()
            self.th_camera.udp_send.disconnect()
            self.cap_camera.release()

    def set_image(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def set_image2(self, image):
        self.label_2.setPixmap(QPixmap.fromImage(image))

    def ping_test(self):
        ip_port_list = self.ip_list
        data = os.system('ping ' + ip_port_list[0][0] + ' -c 2')  # 使用os.system返回值判断是否正常
        if data != 0:
            QtWidgets.QMessageBox.information(self, "Information", "请检查本机网络配置，修改后重新启动本程序！")
        data = os.system('ping ' + ip_port_list[1][0] + ' -c 2')  # 使用os.system返回值判断是否正常
        if data != 0:
            QtWidgets.QMessageBox.information(self, "Information", "请检查RSU的IP网络配置，修改后重新启动本程序！")
        data = os.system('ping ' + ip_port_list[2][0] + ' -c 2')  # 使用os.system返回值判断是否正常
        if data != 0:
            QtWidgets.QMessageBox.information(self, "Information", "请检查摄像头的IP网络配置，修改后打开摄像头！")

    # 获取文本框内容
    def addUL(self):
        print("click addUL")
        global le_ul_x, le_ul_y
        global p1_x, p1_y
        if self.Position_x.text() == "out of range" or self.Position_y.text() == "out of range":
            QtWidgets.QMessageBox.information(self, "Warning", "请选择视频范围内的点")
        else:
            p1_x = self.Position_x.text()
            p1_y = self.Position_y.text()
            le_ul_x = self.LE_UL_x.text()
            le_ul_y = self.LE_UL_y.text()

    def addUR(self):
        print("click addUR")
        global le_ur_x, le_ur_y
        global p2_x, p2_y
        if self.Position_x.text() == "out of range" or self.Position_y.text() == "out of range":
            QtWidgets.QMessageBox.information(self, "Warning", "请选择视频范围内的点")
        else:
            p2_x = self.Position_x.text()
            p2_y = self.Position_y.text()
            le_ur_x = self.LE_UR_x.text()
            le_ur_y = self.LE_UR_y.text()

    def addDL(self):
        print("click addDL")
        global le_dl_x, le_dl_y
        global p3_x, p3_y
        if self.Position_x.text() == "out of range" or self.Position_y.text() == "out of range":
            QtWidgets.QMessageBox.information(self, "Warning", "请选择视频范围内的点")
        else:
            p3_x = self.Position_x.text()
            p3_y = self.Position_y.text()
            le_dl_x = self.LE_DL_x.text()
            le_dl_y = self.LE_DL_y.text()

    def addDR(self):
        print("click addDR")
        global le_dr_x, le_dr_y
        global p4_x, p4_y
        if self.Position_x.text() == "out of range" or self.Position_y.text() == "out of range":
            QtWidgets.QMessageBox.information(self, "Warning", "请选择视频范围内的点")
        else:
            p4_x = self.Position_x.text()
            p4_y = self.Position_y.text()
            le_dr_x = self.LE_DR_x.text()
            le_dr_y = self.LE_DR_y.text()

    def clear_xy(self):
        global p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y
        p1_x = p1_y = p2_x = p2_y = p3_x = p3_y = p4_x = p4_y = 0

#筛选目标，在ROI区域内，删除重叠的目标
def select_obj1(img, output, classes, colors, quadrangle):
    #删除不在ROI区域内的目标
    mask = np.ones(len(output), dtype=np.bool)
    for i, obj in enumerate(output):
        box = obj[1:5].astype(int)
        down_mid_pt = ((box[0] + box[2]) // 2, box[3])
        ul_pt = (quadrangle[0][0], quadrangle[0][1])
        ur_pt = (quadrangle[1][0], quadrangle[1][1])
        dr_pt = (quadrangle[2][0], quadrangle[2][1])
        dl_pt = (quadrangle[3][0], quadrangle[3][1])
        if not is_pt_in_quadrangle(down_mid_pt, ul_pt, ur_pt, dr_pt, dl_pt):
            mask[i] = False

    output = output[mask]
    bboxes = output[:, 1:5]   #bbox
    cls_idx = output[:, -1]    #class_index
    conf = output[:, 5]         #confidence

    # 删除重叠的目标
    mask2 = np.ones(len(bboxes), np.bool)
    if len(bboxes) > 1:
        for i in range(len(bboxes)-1):
            if not mask2[i]:
                continue
            bb1 = bboxes[i]
            for j in range(i+1, len(bboxes)):
                bb2 = bboxes[j]
                iou = calculate_overlap(bb1, bb2)
                if iou > 0.6:
                    if conf[i] > conf[j]:  #重叠的话，保留置信度大的目标
                        # conf[j] = -1
                        mask2[j] = False
                    else:
                        # conf[i] = -1
                        mask2[i] = False
                        break
    conf = conf[mask2]
    bboxes = bboxes[mask2]
    cls_idx = cls_idx[mask2]

    result_list = []
    label_list = []
    for i, x in enumerate(bboxes):
        box = x[0:4].astype(int)
        cls = int(cls_idx[i])
        conf1 = round(float(conf[i]), 2)
        color = colors[cls]
        label = "{0}: {1}".format(classes[cls], conf1)
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, 3)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 1.1, 2)[0]
        c2_new = box[0] + t_size[0] + 6, box[1] + t_size[1] + 18
        cv2.rectangle(img, (box[0], box[1]), c2_new, color, -1)
        cv2.putText(img, label, (box[0], box[1] + t_size[1] + 4), cv2.FONT_HERSHEY_COMPLEX, 1.1, [225, 255, 255], 2)

        down_mid_pt = ((box[0] + box[2]) // 2, box[3])
        v = 0
        v_angle = 0
        real_class = 0

        if cls == 3:
            real_class = 3
        elif cls == 4 or cls == 5:
            real_class = 2
        elif cls == 0 or cls == 1 or cls == 2:
            real_class = 1

        result_list.append([real_class, down_mid_pt, (v, v_angle)])
        label_list.append(classes[cls])

    return result_list, label_list

def deleteOverlap(objs, labels):
    num = len(objs)
    delRes = [0, (-1, -1), (0, 0), []]
    for i in range(0,num-1):
        if objs[i] != delRes:
            box1 = objs[i][-1]
            for j in range(i+1, num):
                if objs[j] == delRes:
                    continue
                box2 = objs[j][-1]
                iou = calculate_overlap(box1[:2], box2[:2])
                if iou > 0.6:
                    if box1[-1] > box2[-1]:
                        objs[j] = delRes
                    else:
                        objs[i] = delRes

    for i in range(0,num):
        if objs[i] == delRes:
            del objs[i]
            del labels[i]

    return objs, labels


def video_demo(frame, CUDA, model, confidence, num_classes, nms_thesh, inp_dim, quadrangle):
    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1, 2)

    if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()

    with torch.no_grad():
        output = model(Variable(img), CUDA)
    output = write_results(output, confidence, num_classes, nms=True, nms_conf=nms_thesh)
    if type(output) == int:
        # end = time.time()
        # seconds = end - start
        # fps = (fps + (1 / seconds)) / 2
        return orig_im, 0, [], []

    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    # target_num = output.shape[0]

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

    classes = load_classes('data/new.names')
    colors = pkl.load(open("pallete", "rb"))
    # 针对性改进__BaiYu write_select 替代 write ，只显示四边形内部检测信息
    # result_list = list(map(lambda x: write_select(x, orig_im, classes, colors, quadrangle), output))
    output = output.cpu().numpy()
    result_list, label_list = select_obj1(orig_im, output, classes, colors, quadrangle)
    # labels = output[:, -1]
    # label_list = []
    # for label in labels:
    #     label_list.append(classes[int(label)])
    # # print("修改前", target_num)
    # # print("修改前", bottom_pt)
    # box_list = []
    # label_list = []
    # for bi in range(len(result_list) - 1, -1, -1):
    #     if result_list[bi][0][1] == (-1, -1):
    #         continue
    #     else:
    #         box_list.append(result_list[bi][0])
    #         label_list.append(result_list[bi][1])
    target_num = len(result_list)
    # if target_num > 1:
    #     box_list, label_list = deleteOverlap(result_list, label_list)
    # if target_num>0:
    #     print(target_num)
    # return orig_im, target_num, result_list, label_list
    return orig_im, target_num, result_list, label_list


def drawM(img, pls):
    pts = pls.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    return img


def draw_pt(img, pls):
    for i in pls:
        ti = tuple(i)
        if ti[0] != 0 and ti[1] != 0:
            img_r = cv2.circle(img, ti, 3, (0, 255, 255), 3)
        else:
            img_r = img
    return img_r


# 采用线程来播放视频
class Thread(QThread):
    change_pix_map = pyqtSignal(QtGui.QImage)
    change_pix_map2 = pyqtSignal(QtGui.QImage)
    update_date = pyqtSignal(str)
    udp_send = pyqtSignal(list)

    def __init__(self):
        super(Thread, self).__init__()
        self.run_or_stop = False

    def run(self):
        # download the net_model
        args = arg_parse()
        confidence = float(args.confidence)
        nms_thesh = float(args.nms_thresh)
        CUDA = torch.cuda.is_available()
        print("CUDA is available:", CUDA)
        num_classes = 6
        print("Loading network.....")
        print(args.weightsfile)
        model = Darknet(args.cfgfile)
        model.load_weights(args.weightsfile)
        print("Network successfully loaded")
        model.net_info["height"] = args.reso
        inp_dim = int(model.net_info["height"])
        assert inp_dim % 32 == 0
        assert inp_dim > 32
        if CUDA:
            model.cuda()
        model.eval()

        time.sleep(1)

        global p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, p4_x, p4_y
        global le_ul_x, le_ul_y, le_ur_x, le_ur_y, le_dl_x, le_dl_y, le_dr_x, le_dr_y
        global Ret, Frame, Get_Flag, count_c
        # start mark and show
        # print(ret)

        # fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
        # out = cv2.VideoWriter('original.avi', fourcc, 10, (Frame.shape[1], Frame.shape[0]))
        # outResult = cv2.VideoWriter('result.avi', fourcc, 10, (Frame.shape[1], Frame.shape[0]))
        # cap = cv2.VideoCapture('original-t.avi')
        # count = 0
        while Get_Flag:
        # while cap.isOpened():
            # Ret, Frame = cap.read()
            if Ret:
                # print(count)
                # count += 1
                frame = Frame
                # out.write(frame)
                # print("count_c =", count_c)
                if self.run_or_stop:
                    print("Thread stop")
                    break
                scale = frame.shape[0] / 630  # 使视频流适应窗口  1120,630
                # print(frame.shape)
                _h = frame.shape[0]
                _w = frame.shape[1]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 如果显示颜色不正常，请自行选择隐藏或不隐藏这一行
                detect_result = np.zeros((150, 400), dtype=np.uint8)  # 用于显示黑色检测目标的情况 width:400 height:150
                pls = np.int32(np.array([[p1_x, p1_y], [p2_x, p2_y], [p4_x, p4_y], [p3_x, p3_y]], np.int32) * scale)
                if [p1_x, p1_y] == [0, 0] or [p2_x, p2_y] == [0, 0] or [p3_x, p3_y] == [0, 0] or [p4_x, p4_y] == [0, 0]:
                    rgb_image = draw_pt(frame, pls)
                else:
                    ic = is_cross(pls)
                    if ic == 1:
                        p3_x, p3_y, p4_x, p4_y = p4_x, p4_y, p3_x, p3_y
                    if ic == 2:
                        p3_x, p3_y, p2_x, p2_y = p2_x, p2_y, p3_x, p3_y
                    mys = time.time()
                    #检测函数
                    # rgb_image, t_n, result_get, label_get = video_demo(frame, CUDA, model, confidence, num_classes, nms_thesh,
                    #                                         inp_dim, pls)
                    rgb_image, t_n, result_get, label_get= video_demo(frame, CUDA, model, confidence, num_classes, nms_thesh, inp_dim, pls)

                    # print("\n\nFPS:", 1 / (time.time() - mys), "\n\n")
                    self.update_date.emit(str(t_n))
                    rgb_image = drawM(rgb_image, pls)

                    result_final = []
                    if t_n != 0:
                        pls_xy = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]], np.float64) * scale
                        mini_xy = np.array([[0, 0], [400, 0], [0, 150], [400, 150]], np.float64)
                        real_xy = np.float64([[le_ul_x, le_ul_y], [le_ur_x, le_ur_y],
                                              [le_dl_x, le_dl_y], [le_dr_x, le_dr_y]])
                        try:
                            M_minibox = get_perspective(pls_xy, mini_xy)
                            M_real = get_perspective(pls_xy, real_xy)
                        except Exception as e:
                            print(e)
                            continue
                        global Sensor_ID
                        # 仅对指定点透视变换计算坐标
                        for i in range(t_n):
                            result_final.append([i])  # 目标ID
                            w_pt_x, w_pt_y = warp_perspective_point(result_get[i][1], M_minibox)
                            r_pt_x, r_pt_y = warp_perspective_point(result_get[i][1], M_real)
                            # r_pt_x = r_pt_y = w_pt_x = w_pt_y =10
                            detect_result = cv2.circle(detect_result, (int(w_pt_x), int(w_pt_y)), 1, 255, 3)
                            result_final[i].append(result_get[i][0])  # 目标类型
                            result_final[i].append(3)  # 目标传感器类型
                            result_final[i].append(Sensor_ID)  # 目标传感器ID
                            result_final[i].append(float('%.7f' % r_pt_x))  # 目标经度
                            result_final[i].append(float('%.7f' % r_pt_y))  # 目标纬度
                            # print(float('%.7f' % r_pt_x))
                            # print(float('%.7f' % r_pt_y))
                            if r_pt_y >= 0:
                                result_final[i].append('N')  # 目标纬度扩展
                            else:
                                result_final[i].append('S')
                            if r_pt_x >= 0:
                                result_final[i].append('E')  # 目标经度扩展
                            else:
                                result_final[i].append('W')
                            result_final[i].append(result_get[i][2][0])  # 目标速度大小
                            result_final[i].append(result_get[i][2][1])  # 目标速度方向

                    # Format_Grayscale8 | Format_RGB888
                    convert_to_qt_format2 = QtGui.QImage(detect_result.data, detect_result.shape[1],
                                                         detect_result.shape[0], QImage.Format_Grayscale8)
                    p2 = convert_to_qt_format2.scaled(400, 150, Qt.KeepAspectRatio)
                    self.change_pix_map2.emit(p2)

                    #out.write(detect_result)  # 保存视频
                    print("类别： ")
                    print(label_get)
                    print("发送结果： ")
                    print(result_final)

                    self.udp_send.emit(result_final)

                #out.release()

                convert_to_qt_format = QtGui.QImage(rgb_image.data, _w, _h, QImage.Format_RGB888)
                p = convert_to_qt_format.scaled(1120, 630, Qt.KeepAspectRatio)
                self.change_pix_map.emit(p)
                self.update_date.emit(str(0))
                # print(time.time())
                rgb_image1 = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                # outResult.write(rgb_image1)

        # outResult.release()
        # out.release()
    def stop(self):
        self.run_or_stop = True


# 采用线程来循环播放视频，获取当前帧图像信息
class FrameReadLoop(QThread):
    warning_message = pyqtSignal(str)

    def __init__(self, cap_name_get, v_or_c):
        super(FrameReadLoop, self).__init__()
        self.cap_name = cap_name_get
        self.cap = None
        self.flag = 0
        self.run_or_stop = False
        self.video_or_cam = v_or_c

    def run(self):
        # read and save
        print(self.cap_name)
        self.cap = cv2.VideoCapture(self.cap_name)
        # self.cap = cv2.VideoCapture('/home/user/yy/workspace/dataSet/jianzhou_data/1/original.avi')
        print(self.cap.isOpened())
        global Ret, Frame, Get_Flag, count_c, cam_IP
        while self.cap.isOpened():
            count_c += 1
            if self.run_or_stop:
                print("FrameReadLoop Thread stop")
                break
            Ret, Frame = self.cap.read()
            count_c += 1
            if Ret:
                if self.video_or_cam == "v":
                    time.sleep(0.04)  # 控制视频播放的速度
                # time.sleep(0.2)
            else:
                print("尝试重新连接摄像头", count_c)
                self.flag += 1
                time.sleep(0.2)
                ping_code = os.system('ping ' + cam_IP + ' -c 2')
                if ping_code != 0:
                    print("摄像头网线断开")
                    self.cap.release()
                    # raise Exception('connect failed.')
                    self.run_or_stop = True
                    # break
                    while self.flag == 1:
                        if CameraFlag == 1:
                            self.cap = cv2.VideoCapture(self.cap_name)
                            Ret, Frame = self.cap.read()
                            print('摄像头网线断开，正尝试链接')
                            time.sleep(1)
                            if Ret != 0:
                                self.flag = 0
                                self.run_or_stop = False
                                break
                        else:
                            break
                    if self.flag == 0:
                        continue
                    else:
                        break
                else:
                    self.cap = cv2.VideoCapture(self.cap_name)
                    if self.flag >= 25:
                        print("尝试失败")
                        self.warning_message.emit('无法获取视频流')
                        Get_Flag = False
                        break
        self.cap.release()
        print("信号消失")

    def stop(self):
        self.run_or_stop = True


if __name__ == '__main__':
    # 加载现有配置文件
    app = QtWidgets.QApplication(sys.argv)
    window = mywindow()
    window.check_conf('NL_config')
    # window.ping_test()
    # window.camera_processing()
    window.video_processing()
    # Thread()
    window.show()
    sys.exit(app.exec_())
