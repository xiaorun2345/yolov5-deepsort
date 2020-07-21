#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ${NAME}.py
@Modify Time      @Author  Xiaorun   @Version    @Desciption
将激光雷达坐标与微波雷达坐标通过PL=RP+T进行转换,其中object_3D传入2D点的列表，输出得到3D点坐标
object_2D传入3D点的列表，输出得到2D点坐标，此程序用来作为不同坐标系之间的转换
'''
import cv2
import numpy as np
import math
import json
class pnp_object_location:
    '''
    导入相机外参
    '''
    def __init__(self,cfg):
        self.cfg=cfg

    def com_cfg(self):
        # 读取json文件，获取各种配置参数
        with open(self.cfg, "r", encoding="utf-8") as fp:
            json_data = json.load(fp)
            #print(json_data)
            image_path = json_data["genInfo"]["inFrmPth"]
            centers2D = json_data["camCal"]["cal2dPtLs"]
            centers3D = json_data["camCal"]["cal3dPtLs"]
            object_2d_points = np.array((centers2D), dtype=np.double)
            object_3d_point = np.array((centers3D), dtype=np.double)
            return object_2d_points,object_3d_point
    ##############################################################
    #此方法为求取平面单应性矩阵
    ###############################################################
    def solve_Hom(self,o2dpts,o3dpts):
        h, status = cv2.findHomography(o3dpts,o2dpts, 4, 0)
        h_inv, status = cv2.findHomography(o2dpts, o3dpts, 4, 0)
        return h, h_inv

    #2d到3d的映射
    def object_2to3D(self,centers,homegraphy):
        '''

        :param centers:
        :param Num:
        :return:
       将n个2D坐标转换为3D坐标
        '''
        centers3D = []
        for center in centers:
            o3dPtMat=np.ones((3,1))
            o3dPtMat[0]=center[0]
            o3dPtMat[1]=center[1]
            center3D=np.matmul(homegraphy,o3dPtMat)
            opt3D=(center3D[0]/center3D[2],center3D[1]/center3D[2],1)
            centers3D.append(opt3D)
        return centers3D

    def object_3to2D(self,centers,homegraphy_inv):
        '''
        :param self:
        :param centers:
        :param Num:
        :return:
        将n个3D坐标转换为3D坐标
        '''
        centers2D=[]
        for center  in centers:
            o2dptMat=np.ones((3,1))
            o2dptMat[0]=center[0]
            o2dptMat[1]=center[1]
            center2D=np.matmul(homegraphy_inv, o2dptMat)
            opt2D =(center2D[0]/center2D[2],center2D[1]/center2D[2])
            centers2D.append(opt2D)
        return centers2D

if __name__ == '__main__':
    path = "cfg_lidar_camera.json"
    solve_homegraphy = pnp_object_location(path)
    object_2d_points, object_3d_point = solve_homegraphy.com_cfg()
    h, h_inv = solve_homegraphy.solve_Hom(object_2d_points, object_3d_point)

    print('h = ')
    print(h)
    print('h_inv = ')
    print(h_inv)