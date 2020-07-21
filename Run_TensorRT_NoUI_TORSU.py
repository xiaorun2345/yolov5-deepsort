# -*- coding: utf-8 -*-
from logger import logger
import traceback
try:
    import threading
    from OnnxTensorrtModule import *
    from deep_sort import DeepSort
    import os
    import subprocess as sp
    from parse_data import parse_conf, parse_argument
    from data_sender import *
    from detection import *
    from datetime import datetime
    from lidar_camera_utils import pnp_object_location
except:
    logger.error(traceback.format_exc())
global Frame
Frame = []

global isRunning, is_show
isRunning = is_show = True

global Ret
Ret = 0

HeartBeatTime = 20  #心跳发送时间间隔，单位s

# 采用线程来循环播放视频，获取当前帧图像信息
class FrameReadLoop(threading.Thread):
    def __init__(self, _capName, _camIP):
        super(FrameReadLoop, self).__init__()
        self.capName = _capName
        self.flag = 0
        self.idx = _camIP
        # self.Sensor_ID = "original"

    def run(self):
        global Ret, Frame, isRunning
        logger.info('frame read thread is starting...')

        cap = cv2.VideoCapture(self.capName)

        while isRunning:
            start = time.time()
            Ret, Frame = cap.read()
            if not Ret:
                self.flag += 1
                logger.info('try to cap the camera')
                cap = cv2.VideoCapture(self.capName)
                if self.flag > 25:
                    logger.info('try failure')
                    break
                time.sleep(0.01)

        isRunning = False
        cap.release()
        logger.info("FrameReadLoop thread stop.")

class CameraHeartbeatThread(threading.Thread):
    def __init__(self, _heartIP, _heartPort, _camIP):
        super(CameraHeartbeatThread, self).__init__()
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.heartIP = _heartIP
        self.heartPort = _heartPort
        self.cam_IP = _camIP
        self.flag = 1

    def run(self):
        global isRunning
        print('camera heart beat thread is starting...')
        print('cam_ip:%s  cam_heart_ip:%s,  cam_heart_port:%d' % (self.cam_IP, self.heartIP, self.heartPort))
        while 1:
            isBroken = False  # 有无故障  1：故障  2：正常
            errorNumber = b'\x01'  # 故障代码  1网络不通  2数据异常  3 检测程序退出
            heartPackage = []

            ping_code = os.system('ping ' + self.cam_IP + ' -c 2')
            if ping_code != 0:
                isBroken = True
                errorNumber = b'\x01'

            if not isRunning:
                isBroken = True
                errorNumber = b'\x03'
                self.flag += 25
                if self.flag > 25:
                    break

            if isBroken:
                heartPackage.append(b'\x01')
                heartPackage.append(errorNumber)
            else:
                heartPackage.append(b'\x02')
            print(heartPackage)

            udp_sender(self.socket, (self.heartIP, self.heartPort), heartPackage)

            time.sleep(HeartBeatTime)
        logger.info('heartbeat thread stop.')

def cameraDetectionDemo(dim, pls, real_xy, jsonName, sensorID, udpClient, addr, camIP):
    global isRunning, Frame, is_show
    onnx_file_path = 'weights/yolov3-416-new_best.onnx'
    engine_file_path = "weights/yolov3-416-new_best.trt"
    init_dict = {'trt': engine_file_path, 'onnx': onnx_file_path}
    onnx2trt = OnnxTensorrtModule(init_dict)

    classes = load_classes('data/new.names')
    colors = pkl.load(open("pallete", "rb"))

    # jsonName = 'cfg.json'
    solve_homegraphy = pnp_object_location(jsonName)
    object_2d_points, object_3d_point = solve_homegraphy.com_cfg()
    h, h_inv = solve_homegraphy.solve_Hom(object_2d_points, object_3d_point)

    scale = 1080 / 630
    pls = pls *scale
    M_real = get_perspective(pls, real_xy)

    ic = is_cross(pls)
    if ic == 1:
        tmp = [pls[2][0], pls[2][1]]
        pls[2], pls[3] = pls[3], tmp
    if ic == 2:
        tmp = [pls[2][0], pls[2][1]]
        pls[2], pls[1] = pls[1], tmp
    pls = pls.astype(np.int)

    deepsort = DeepSort('./deep_sort/deep/checkpoint/ckpt.t7')

    is_show = False
    if is_show:
        cv2.namedWindow(sensorID, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(sensorID, (900,600))

    frameNo = 1
    while isRunning:
        start = time.time()

        detect_time = time.time()
        if Ret and Frame is not None:
            frame = Frame

            detect_time = time.time()
            rgb_image, outputs_tracking = video_demo(frame, dim, pls, onnx2trt, deepsort, classes, colors, h_inv)

            detect_time = time.time() - detect_time
            detect_time *= 1000
            print(detect_time)

            resultSender(outputs_tracking, h_inv, Sensor_ID, udpClient, addr, detect_time)
            rgb_image = drawM(rgb_image, pls)

            if is_show:
                cv2.imshow(sensorID, rgb_image)
                if cv2.waitKey(1) == 27:
                    break
            # time.sleep(0.05)

            frameNo += 1

            end = max(1, (time.time() - start)*1000)
            s = '{}: detect:{:.2f} ms total:{:.2f} ms fps:{:.1f}'.format(frameNo, detect_time, end, 1000/end)
            # print(s)
            # print(frameNo)

    isRunning = False
    if is_show:
        cv2.destroyWindow(sensorID)
    # cap.release()

if __name__ == '__main__':
    print('*****************************************')
    print('version1.2: send message to rsu')
    print('*****************************************')
    try:
        args = parse_argument()
        idx = args.idx
        Sensor_ID, local_IP, local_port, heart_port, cameraAddr, cam_IP, pls_xy, real_xy, RSU_IP, RSU_port, jsonName, is_show = parse_conf(
            'NL_config', idx)
        udpClient, addr = udp_client(RSU_IP, RSU_port)

        # cameraAddr = '/home/user/yy/workspace/dataSet/jianzhou_data/3/out.avi'
        # cameraAddr = '/home/user/yy/workspace/dataSet/jianzhou_data/1/out.avi'
        cameraAddr = '/home/nvidia/yy/nebula/20190911-1824.avi'
        t_get_frame_from_stream = FrameReadLoop(cameraAddr, cam_IP)
        t_get_frame_from_stream.start()
        # t_get_frame_from_stream.join()

        cameraDetectionDemo(416, pls_xy, real_xy, jsonName, Sensor_ID, udpClient, addr, cam_IP)

    except:
        logger.error(traceback.format_exc())
        isRunning = False

