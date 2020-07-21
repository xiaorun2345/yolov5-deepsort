import perceptron_pb2
import socket
import struct
import time
# from detection import warp_perspective_point
import numpy as np
from Point2GPS import *

def udp_client(host, port):
    # global local_IP
    # print(local_IP)
    # host = '192.168.110.196'  #local_IP  # 这是客户端的电脑的ip
    # host = local_IP
    # port = 8894  #13141  # 接口选择大于10000的，避免冲突
    # bufsize = 1024  # 定义缓冲大小
    addrSender = (host, port)  # 元组形式
    udpClientSender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建客户端

    return udpClientSender, addrSender

def udp_sender_protobuf(udpClient, addr, data_protobuf):
    # global udpClient, addr
    data_s = ''
    data_head = b'\xda\xdb\xdc\xdd'
    data_s += data_head.hex()
    frame_type = b'\x01'
    data_s += frame_type.hex()
    perception_type = b'\x00'
    data_s += perception_type.hex()

    target_number = len(data_protobuf)
    target_number_b = struct.pack('!h', target_number)
    data_s += target_number_b.hex()
    data_s += data_protobuf.hex()
    data = bytes.fromhex(data_s)
    # print(data)
    udpClient.sendto(data, addr)

def udp_sender(udpClient, addr, data):
    # global udpClient, addr
    data_s = ''
    data_head = b'\xda\xdb\xdc\xdd'
    data_s += data_head.hex()
    frame_type = b'\x01'
    data_s += frame_type.hex()
    perception_type = b'\x00'
    data_s += perception_type.hex()

    target_number = len(data)
    target_number_b = struct.pack('!h', target_number)
    data_s += target_number_b.hex()

    for d in data:
        data_s += d.hex()
    data = bytes.fromhex(data_s)
    udpClient.sendto(data, addr)

def resultSender(outputs_tracking, h_inv, Sensor_ID, udpClient, addr, timestamp):
    perceptronList = perceptron_pb2.PerceptronSet()
    if outputs_tracking is not None and len(outputs_tracking) > 0:

        for track in outputs_tracking:
            perceptron = perceptronList.perceptron.add()
            # global cam_IP
            perceptron.Devide_Id = Sensor_ID  # 可以填摄像机的ip地址
            # optional string Devide_Id = 1;  // 设备号.
            perceptron.Devide_Is_True = True  # 判断设备数据的有效性.
            perceptron.Number_Frame = 0  # 帧的序列号.
            # UNKNOWN = 0; // 未知
            # PEDESTRIAN = 1; // 人
            # BICYCLE = 2; // 非机动车
            # CAR = 3; // 小汽车
            # TRUCK_BUS = 4; // 大车

            if track[4] == 3:     #这个类别按照real_class来 跟踪的时候也按照这个类别
                perceptron.Object_Class_Type = perceptron_pb2.Perceptron.PEDESTRIAN  # 目标种类.
            elif track[4] == 4 or track[4] == 5:
                perceptron.Object_Class_Type = perceptron_pb2.Perceptron.BICYCLE  # 目标种类.


            perceptron.Object_Id = track[5]  # 目标Id.

            if len(track[-1]) >= 10:
                object_ceter = np.ones((3, 1))
                object_ceter[0] = track[-1][9][0]
                object_ceter[1] = track[-1][9][1]
                after = np.matmul(h_inv, object_ceter)
                after[0] = after[0] / after[2]
                after[1] = after[1] / after[2]
                after[2] = 1
                after_longitude, after_latitude = after[0], after[1]

                object_ceter[0] = track[-1][0][0]
                object_ceter[1] = track[-1][0][1]
                before = np.matmul(h_inv, object_ceter)
                before[0] = before[0] / before[2]
                before[1] = before[1] / before[2]
                before[2] = 1
                before_longitude, before_latitude = before[0], before[1]

                speed, heading = GPS2Speed(before_longitude, before_latitude, after_longitude, after_latitude, timestamp * 10)
                perceptron.Object_Speed = speed
                perceptron.Object_Yaw = heading


            perceptron.Point3f.x = (track[0] + track[2]) / 2.0  # 目标的几何中心，单位为米.  用bbox centerX centerY width height
            perceptron.Point3f.y = (track[1] + track[3]) / 2.0  # 目标的几何中心，单位为米.
            perceptron.Point3f.z = 1
            # optional float Object_Confidence = 7;  // 目标置信度.
            perceptron.Object_Confidence = 0
            # optional float Object_Speed = 8; // 目标速度.
            # perceptron.Object_Speed = 0
            # optional float Object_acceleration = 9;  // 目标加速度.
            perceptron.Object_acceleration = 0
            perceptron.Object_Width = track[2] - track[0]  # 目标宽度.  用bbox centerX centerY width height

            perceptron.Object_Length = 0  # 目标长度.
            perceptron.Object_Height = track[3] - track[1]  # 目标高度. 用bbox centerX centerY width height

            down_mid_pt = (int((track[0] + track[2]) / 2.0), int(track[3]))

            # r_pt_x, r_pt_y = warp_perspective_point(down_mid_pt, M_real)
            # o3dPtMat = np.ones((3, 1))
            # o3dPtMat[0] = down_mid_pt[0]
            # o3dPtMat[1] = down_mid_pt[1]
            o3dPtMat = np.array([[down_mid_pt[0]], [down_mid_pt[1]], [1]])
            center3D = np.matmul(h_inv, o3dPtMat)
            r_pt_x = center3D[0] / center3D[2]
            r_pt_y = center3D[1] / center3D[2]

            # optional double Object_Longitude = 13; // 经度.
            perceptron.Object_Longitude = float('%.8f' % r_pt_x)
            # optional double Object_Latitude = 14; // 维度.
            perceptron.Object_Latitude = float('%.8f' % r_pt_y)
            # optional NS Object_NS = 15; // 南北纬.
            # optional EW Object_WE = 16; // 东西经.
            perceptron.Time_stamp = int(time.time())  # 时间戳

            # text = "{0:.2f},{1:.2f}".format(r_pt_x, r_pt_y)
            # t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)[0]
            # c2_new = max(0, down_mid_pt[0] - t_size[0] // 2), down_mid_pt[1]
            # cv2.putText(rgb_image, text, c2_new, cv2.FONT_HERSHEY_COMPLEX, 1.2, [255, 0, 0], 2)
        # print(perceptronList.perceptron)
        data_perceptron = perceptronList.SerializeToString()
        # udpClient.sendto(data_perceptron, addr)  # 不加包头发送数据
        udp_sender_protobuf(udpClient, addr, data_perceptron)

def fusionResultSender(outputs_tracking, M_real, Sensor_ID, udpClient, addr, timestamp):
    global h, h_inv
    perceptronList = perceptron_pb2.PerceptronSet()
    if outputs_tracking is not None and len(outputs_tracking) > 0:

        for track in outputs_tracking:
            perceptron = perceptronList.perceptron.add()
            # global cam_IP
            perceptron.Devide_Id = Sensor_ID  # 可以填摄像机的ip地址
            # optional string Devide_Id = 1;  // 设备号.
            perceptron.Devide_Is_True = True  # 判断设备数据的有效性.
            perceptron.Number_Frame = 0  # 帧的序列号.
            # UNKNOWN = 0; // 未知
            # PEDESTRIAN = 1; // 人
            # BICYCLE = 2; // 非机动车
            # CAR = 3; // 小汽车
            # TRUCK_BUS = 4; // 大车
            # if track[4] == 0:
            #     perceptron.Object_Class_Type = perceptron_pb2.Perceptron.PEDESTRIAN  # 目标种类.
            # elif track[4] == 1 or track[4] == 3:
            #     perceptron.Object_Class_Type = perceptron_pb2.Perceptron.BICYCLE  # 目标种类.

            if track[4] == 3:     #这个类别按照real_class来 跟踪的时候也按照这个类别
                perceptron.Object_Class_Type = perceptron_pb2.Perceptron.PEDESTRIAN  # 目标种类.
            elif track[4] == 4 or track[4] == 5:
                perceptron.Object_Class_Type = perceptron_pb2.Perceptron.BICYCLE  # 目标种类.


            perceptron.Object_Id = track[5]  # 目标Id.

            if len(track[-1]) >= 10:
                object_ceter = np.ones((3, 1))
                object_ceter[0] = track[-1][9][0]
                object_ceter[1] = track[-1][9][1]
                after = np.matmul(h_inv, object_ceter)
                after[0] = after[0] / after[2]
                after[1] = after[1] / after[2]
                after[2] = 1
                after_longitude, after_latitude = after[0], after[1]

                object_ceter[0] = track[-1][0][0]
                object_ceter[1] = track[-1][0][1]
                before = np.matmul(h_inv, object_ceter)
                before[0] = before[0] / before[2]
                before[1] = before[1] / before[2]
                before[2] = 1
                before_longitude, before_latitude = before[0], before[1]

                speed, heading = GPS2Speed(before_longitude, before_latitude, after_longitude, after_latitude, timestamp * 10)
                perceptron.Object_Speed = speed
                perceptron.Object_Yaw = heading


            perceptron.Point2f.x = (track[0] + track[2]) / 2.0  # 目标的几何中心，单位为米.  用bbox centerX centerY width height
            perceptron.Point2f.y = (track[1] + track[3]) / 2.0  # 目标的几何中心，单位为米.
            # optional float Object_Confidence = 7;  // 目标置信度.
            perceptron.Object_Confidence = 0
            # optional float Object_Speed = 8; // 目标速度.
            # perceptron.Object_Speed = 0
            # optional float Object_acceleration = 9;  // 目标加速度.
            perceptron.Object_acceleration = 0
            perceptron.Object_Width = track[2] - track[0]  # 目标宽度.  用bbox centerX centerY width height

            perceptron.Object_Length = 0  # 目标长度.
            perceptron.Object_Height = track[3] - track[1]  # 目标高度. 用bbox centerX centerY width height

            down_mid_pt = (int((track[0] + track[2]) / 2.0), int(track[3]))

            # r_pt_x, r_pt_y = warp_perspective_point(down_mid_pt, M_real)
            o3dPtMat = np.ones((3, 1))
            o3dPtMat[0] = down_mid_pt[0]
            o3dPtMat[1] = down_mid_pt[1]
            center3D = np.matmul(h_inv, o3dPtMat)
            r_pt_x = center3D[0] / center3D[2]
            r_pt_y = center3D[1] / center3D[2]

            # optional double Object_Longitude = 13; // 经度.
            perceptron.Object_Longitude = float('%.8f' % r_pt_x)
            # optional double Object_Latitude = 14; // 维度.
            perceptron.Object_Latitude = float('%.8f' % r_pt_y)
            # optional NS Object_NS = 15; // 南北纬.
            # optional EW Object_WE = 16; // 东西经.
            perceptron.Time_stamp = int(time.time())  # 时间戳

            # text = "{0:.2f},{1:.2f}".format(r_pt_x, r_pt_y)
            # t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1.2, 2)[0]
            # c2_new = max(0, down_mid_pt[0] - t_size[0] // 2), down_mid_pt[1]
            # cv2.putText(rgb_image, text, c2_new, cv2.FONT_HERSHEY_COMPLEX, 1.2, [255, 0, 0], 2)
        # print(perceptronList.perceptron)
        data_perceptron = perceptronList.SerializeToString()
        # udpClient.sendto(data_perceptron, addr)  # 不加包头发送数据
        udp_sender_protobuf(udpClient, addr, data_perceptron)

def cameraSensorList(camera_sensor_list, outputs_tracking, Sensor_ID):
    # camera_sensor_list = []
    # global camera_sensor_list
    if outputs_tracking is not None and len(outputs_tracking) > 0:
        # t_camera_lock.acquire()
        camera_sensor_list = []
        for track in outputs_tracking:
            # global cam_IP
            standard_data = {}
            standard_data['Devide_Id'] = Sensor_ID   # 配置文件中的Sensor_ID
            standard_data['Devide_Is_True'] = True
            standard_data['Number_Frame'] = 0

            if track[4] == 3:
                standard_data['Object_Class_Type'] = perceptron_pb2.Perceptron.PEDESTRIAN
            elif track[4] == 4 or track[4] == 5:
                standard_data['Object_Class_Type'] = perceptron_pb2.Perceptron.BICYCLE
            elif track[4] == 0:
                standard_data['Object_Class_Type'] = perceptron_pb2.Perceptron.CAR
            elif track[4] == 1 or track[4] == 2:
                standard_data['Object_Class_Type'] = perceptron_pb2.Perceptron.TRUCK_BUS

            standard_data['Object_Id'] = track[5]  # 目标Id.
            standard_data['Point2f'] = [(track[0] + track[2]) / 2.0, (track[1] + track[3]) / 2.0]  #目标几何中心  用bbox centerX centerY width height
            standard_data['Object_Confidence'] = track[5]  #confidence
            standard_data['Object_Speed'] = 0
            standard_data['Object_acceleration'] = 0
            standard_data['Object_Width'] = track[2] - track[0]
            standard_data['Object_Length'] = 0
            standard_data['Object_Height'] = track[3] - track[1]

            # down_mid_pt = (int((track[0] + track[2]) / 2.0), int(track[3]))
            # r_pt_x, r_pt_y = warp_perspective_point(down_mid_pt, M_real)
            # standard_data['Object_Longitude'] = float('%.8f' % r_pt_x)
            # standard_data['Object_Latitude'] = float('%.8f' % r_pt_y)

            standard_data['Object_Longitude'] = 0
            standard_data['Object_Latitude'] = 0
            standard_data['Time_stamp'] = int(time.time())
            standard_data['Object_Direction'] = 0
            standard_data['Object_Yaw'] = 0
            standard_data['is_tracker'] = 0     #???
            standard_data['Object_Direction'] = 0

            camera_sensor_list.append(standard_data)
        # t_camera_lock.release()
    return  camera_sensor_list