# -- coding: utf-8 --
# from datapoint import DataPoint,Is_working
# import socket
import perceptron_pb2
import argparse
import configparser
import os
import numpy as np

def parse_Sensor_data(target_data):
    standard_data={}
    standard_data['Devide_Id']=target_data.Devide_Id
    standard_data['Devide_Is_True'] = target_data.Devide_Is_True
    standard_data['Number_Frame'] = target_data.Number_Frame
    standard_data['Object_Class_Type'] = target_data.Object_Class_Type
    # if target_data.Devide_Id == devide_id_name[0]:
    #     standard_data['Object_Id'] = (target_data.Object_Id%10000)+10000
    # if target_data.Devide_Id == devide_id_name[1]:
    #     standard_data['Object_Id'] = (target_data.Object_Id%20000)+20000
    # if target_data.Devide_Id == devide_id_name[2]:
    #     standard_data['Object_Id'] = (target_data.Object_Id%30000)+30000
    # if target_data.Devide_Id == devide_id_name[3]:
    #     standard_data['Object_Id'] = (target_data.Object_Id%40000)+40000
    standard_data['Object_Id'] = target_data.Object_Id
    # standard_data['Point2f'] = target_data.Point2f
    standard_data['Object_Confidence'] = target_data.Object_Confidence
    standard_data['Object_Speed'] = target_data.Object_Speed
    standard_data['Object_acceleration'] = target_data.Object_acceleration
    standard_data['Object_Width'] = target_data.Object_Width
    standard_data['Object_Length'] = target_data.Object_Length
    standard_data['Object_Height'] = target_data.Object_Height
    standard_data['Object_Longitude'] = target_data.Object_Longitude
    standard_data['Object_Latitude'] = target_data.Object_Latitude
    standard_data['Time_stamp'] = target_data.Time_stamp
    standard_data['Object_Direction'] = target_data.Object_Direction
    standard_data['Object_Yaw'] = target_data.Object_Yaw
    standard_data['is_tracker'] = target_data.is_tracker
    # standard_data['Object_Direction'] = target_data.Object_Direction
    return standard_data

#解析收到的protobuf数据，将数据和头信息分开
def parse_message(recv_data):
    head=recv_data[:4]
    #print (head)
    FrameType=recv_data[4]
    PerceptronType=recv_data[5]
    DataLength=recv_data[6:8]
    #print(DataLength)
    DataLength=int.from_bytes(DataLength,byteorder='big',signed=False)
    #print(DataLength)
    Sensor_data=recv_data[8:8+DataLength]
    return head, FrameType, PerceptronType, DataLength, Sensor_data

#解析心跳包数据
def parse_heart_beat(recv_data):
    IsBroken=recv_data[0]
    ErrorNumber=recv_data[1]
    return IsBroken, ErrorNumber

def print_data(object_data):
    print("time_stamp is ", object_data.Time_stamp)
    print("Devide_Id is ", object_data.Devide_Id)
    print("Devide_Is_True is ", object_data.Devide_Is_True)
    print("Number_Frame is ", object_data.Number_Frame)
    print("Object_Class_Type is ", object_data.Object_Class_Type)
    print("Object_Id is ", object_data.Object_Id)
    print("Point2f is ", object_data.Point2f)
    print("Object_Confidence is ", object_data.Object_Confidence)
    print("Object_Speed is ", object_data.Object_Speed)
    print("Object_acceleration is ", object_data.Object_acceleration)
    print("Object_Width is ", object_data.Object_Width)
    print("Object_Length is ", object_data.Object_Length)
    print("Object_Height is ", object_data.Object_Height)
    print("Object_Longitude is ", object_data.Object_Longitude)
    print("Object_Latitude is ", object_data.Object_Latitude)
    print("Object_Direction is ", object_data.Object_Direction)
    print("Object_Yaw is ", object_data.Object_Yaw)
    print("is_tracker is ", object_data.is_tracker)


def parse_data1(Source_data):
    """
    :param Sensor_data:
    字典：三个线程传过来的的传感器数据
    lidar数据，rader数据，摄像头数据
    :return:
    每个传感器数据单一属性
    """
    Sensor_data={}
    for k, v in enumerate(Source_data):
        Sensor_data.update({k : v})

    return  Sensor_data

#将字典数据序列化
def dict_protobuf(sensor_dict_data):
    perceptronList = perceptron_pb2.PerceptronSet()
    # print(sensor_dict_data)
    for obj in range(len(sensor_dict_data)):
        perceptron_objects = perceptronList.perceptron.add()
        object_data = perceptron_objects
        object_data.Devide_Id = sensor_dict_data[obj]['Devide_Id']
        object_data.Devide_Is_True = sensor_dict_data[obj]['Devide_Is_True']
        object_data.Number_Frame = sensor_dict_data[obj]['Number_Frame']
        object_data.Object_Class_Type= sensor_dict_data[obj]['Object_Class_Type']
        object_data.Object_Id = sensor_dict_data[obj]['Object_Id']
        object_data.Point2f.x = sensor_dict_data[obj]['Point2f'][0]
        object_data.Point2f.y = sensor_dict_data[obj]['Point2f'][1]
        object_data.Object_Confidence = sensor_dict_data[obj]['Object_Confidence']
        object_data.Object_Speed = sensor_dict_data[obj]['Object_Speed']
        object_data.Object_acceleration =sensor_dict_data[obj]['Object_acceleration']
        object_data.Object_Width = sensor_dict_data[obj]['Object_Width']
        object_data.Object_Length = sensor_dict_data[obj]['Object_Length']
        object_data.Object_Height =sensor_dict_data[obj]['Object_Height']
        object_data.Object_Longitude = sensor_dict_data[obj]['Object_Longitude']
        object_data.Object_Latitude = sensor_dict_data[obj]['Object_Latitude']
        object_data.Time_stamp = sensor_dict_data[obj]['Time_stamp']
        object_data.Object_Direction = sensor_dict_data[obj]['Object_Direction']
        object_data.Object_Yaw = sensor_dict_data[obj]['Object_Yaw']
        object_data.is_tracker = sensor_dict_data[obj]['is_tracker']
        object_data.Object_Direction = sensor_dict_data[obj]['Object_Direction']
    data_perceptron = perceptronList.SerializeToString()
    return data_perceptron

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", dest="idx", help="process no.", default="1", type=str)
    return parser.parse_args()

def parse_conf_fusion(filename, idx):  # 读取配置文件参数
    conf = configparser.ConfigParser()

    iniName = filename + idx + ".ini"
    print('load configuration from ' + iniName)
    assert (os.path.exists(iniName))
    conf.read(iniName)

    local_IP = conf.get(filename, "local_IP")
    local_port = int(conf.get(filename, 'local_port'))
    localMsg = [local_IP, local_port]

    cam_device_id = conf.get(filename, "cam_device_id")
    cam_heart_port = int(conf.get(filename, 'cam_heart_port'))
    cam_ip = conf.get(filename, "cam_ip")
    cam_port = conf.get(filename, 'cam_port')
    cam_username = conf.get(filename, 'cam_username')
    cam_password = conf.get(filename, 'cam_password')
    cameraAddr = "rtsp://" + cam_username + ":" + cam_password + "@" + cam_ip + ":" + cam_port + "//Streaming/Channels/1"
    cameraMsg = [cam_device_id, cam_ip, cam_heart_port, cameraAddr]

    RSU_IP = conf.get(filename, "rsu_ip")
    RSU_port = int(conf.get(filename, 'rsu_port'))
    rsuMsg = [RSU_IP, RSU_port]

    radar_device_id = conf.get(filename, 'radar_device_id')
    radar_ip = conf.get(filename, "radar_ip")
    radar_port = int(conf.get(filename, 'radar_port'))
    radar_heart_port = int(conf.get(filename, 'radar_heart_port'))
    radarMsg = [radar_device_id, radar_ip, radar_port, radar_heart_port]

    lidar_device_id = conf.get(filename, 'lidar_device_id')
    lidar_ip = conf.get(filename, 'lidar_ip')
    lidar_port = int(conf.get(filename, 'lidar_port'))
    lidar_heart_port = int(conf.get(filename, 'lidar_heart_port'))
    lidarMsg = [lidar_device_id, lidar_ip, lidar_port, lidar_heart_port]

    le_ul_x = conf.get(filename, "le_ul_x")
    le_ul_y = conf.get(filename, "le_ul_y")
    le_ur_x = conf.get(filename, "le_ur_x")
    le_ur_y = conf.get(filename, "le_ur_y")
    le_dl_x = conf.get(filename, "le_dl_x")
    le_dl_y = conf.get(filename, "le_dl_y")
    le_dr_x = conf.get(filename, "le_dr_x")
    le_dr_y = conf.get(filename, "le_dr_y")
    p1_x = int(conf.get(filename, "p1_x"))
    p1_y = int(conf.get(filename, "p1_y"))
    p2_x = int(conf.get(filename, "p2_x"))
    p2_y = int(conf.get(filename, "p2_y"))
    p3_x = int(conf.get(filename, "p3_x"))
    p3_y = int(conf.get(filename, "p3_y"))
    p4_x = int(conf.get(filename, "p4_x"))
    p4_y = int(conf.get(filename, "p4_y"))

    is_show = bool(int(conf.get(filename, 'is_show')))

    # pls = np.int32([[p1_x, p1_y], [p2_x, p2_y], [p4_x, p4_y], [p3_x, p3_y]], np.int32))
    pls_xy = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]], np.float)
    real_xy = np.float64([[le_ul_x, le_ul_y], [le_ur_x, le_ur_y], [le_dl_x, le_dl_y], [le_dr_x, le_dr_y]])

    return localMsg, rsuMsg, cameraMsg, radarMsg, lidarMsg, pls_xy, real_xy, is_show

def str2numpy(str):
    h = []
    # str = str.strip()
    str = str.split(';')
    for s in str:
        s = s.split(' ')
        h.append(s)
    h = np.array(h)
    h = h.astype(np.float)
    return h


def parse_conf(filename, idx):  # 读取配置文件参数
    conf = configparser.ConfigParser()
    global cam_IP
    iniName = filename + idx + ".ini"
    print('load configuration from ' + iniName)
    assert (os.path.exists(iniName))
    conf.read(iniName)

    Sensor_ID = conf.get(filename, "cam_sensor_id")
    local_IP = conf.get(filename, "local_IP")
    local_port = int(conf.get(filename, 'local_port'))
    heart_port = int(conf.get(filename, 'heart_port'))
    RSU_IP = conf.get(filename, "RSU_IP")
    RSU_port = int(conf.get(filename, 'RSU_port'))
    cam_IP = conf.get(filename, "cam_IP")
    cam_port = conf.get(filename, 'cam_port')
    cam_username = conf.get(filename, 'cam_username')
    cam_password = conf.get(filename, 'cam_password')
    le_ul_x = conf.get(filename, "le_ul_x")
    le_ul_y = conf.get(filename, "le_ul_y")
    le_ur_x = conf.get(filename, "le_ur_x")
    le_ur_y = conf.get(filename, "le_ur_y")
    le_dl_x = conf.get(filename, "le_dl_x")
    le_dl_y = conf.get(filename, "le_dl_y")
    le_dr_x = conf.get(filename, "le_dr_x")
    le_dr_y = conf.get(filename, "le_dr_y")
    p1_x = int(conf.get(filename, "p1_x"))
    p1_y = int(conf.get(filename, "p1_y"))
    p2_x = int(conf.get(filename, "p2_x"))
    p2_y = int(conf.get(filename, "p2_y"))
    p3_x = int(conf.get(filename, "p3_x"))
    p3_y = int(conf.get(filename, "p3_y"))
    p4_x = int(conf.get(filename, "p4_x"))
    p4_y = int(conf.get(filename, "p4_y"))

    jsonName = conf.get(filename, 'jsonName')

    # h_str = conf.get(filename, 'h')
    # h = str2numpy(h_str)

    is_show = bool(int(conf.get(filename, 'show')))

    # pls = np.int32([[p1_x, p1_y], [p2_x, p2_y], [p4_x, p4_y], [p3_x, p3_y]], np.int32))
    pls_xy = np.array([[p1_x, p1_y], [p2_x, p2_y], [p3_x, p3_y], [p4_x, p4_y]], np.float)
    real_xy = np.float64([[le_ul_x, le_ul_y], [le_ur_x, le_ur_y], [le_dl_x, le_dl_y], [le_dr_x, le_dr_y]])
    cameraAddr = "rtsp://" + cam_username + ":" + cam_password + "@" + cam_IP + ":" + cam_port + "//Streaming/Channels/1"

    return Sensor_ID, local_IP, local_port, heart_port, cameraAddr, cam_IP, pls_xy, real_xy, RSU_IP, RSU_port, jsonName, is_show
