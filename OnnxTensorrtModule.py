# -*- coding: utf-8 -*-

from __future__ import print_function

from base_module import BaseModule
import torch    # must import otherwise there is an error to translate onnx to trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt
import time
from util2 import *

from data_processing import PreprocessYOLO

import sys, os

sys.path.insert(1, os.path.join(sys.path[0], ".."))
import common

TRT_LOGGER = trt.Logger()


def get_engine(onnx_file_path, engine_file_path=""):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network,
                                                                                                     TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 30  # 1GB
            builder.max_batch_size = 1
            # builder.fp16_mode = True
            # builder.strict_type_constraints = True

            # Parse model file
            if not os.path.exists(onnx_file_path):
                print('ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                raise FileExistsError(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))

            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                parser.parse(model.read())
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            engine = builder.build_cuda_engine(network)

            if engine is None:
                print('build engine have some error')
                raise Exception('build engine have some error')

            with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
                print("Completed creating Engine")
            return engine

    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()

class OnnxTensorrtModule(BaseModule):
    def __init__(self, init_dict):
        a = torch.cuda.FloatTensor()  # pytorch必须首先占用部分CUDA

        self.trt_file = init_dict['trt']
        self.onnx_file = init_dict['onnx']
        self.use_cuda = True
        self.inp_dim = 416  # yolov3-416
        # self.inp_dim = 608  #yolov3-608
        self.num_classes = 6
        # self.output_shapes = [(1, 255, 13, 13), (1, 255, 26, 26), (1, 255, 52, 52)]  # yolov3-416
        self.output_shapes = [(1, 33, 13, 13), (1, 33, 26, 26), (1, 33, 52, 52)]  # yolov3-416
        # self.output_shapes = [(1, 255, 19, 19), (1, 255, 38, 38), (1, 255, 76, 76)] #yolo3-608

        # self.yolo_anchors = [[(116, 90), (156, 198), (373, 326)],
        #                      [(30, 61), (62, 45), (59, 119)],
        #                      [(10, 13), (16, 30), (33, 23)]]

        self.yolo_anchors = [[(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)],
                             [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],
                             [(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)]]

        self.engine = get_engine(self.onnx_file, self.trt_file)
        self.inputs, self.outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        self.context = self.engine.create_execution_context()

    def process_frame(self, frame_dic):
        pass

    def process_frame_batch(self, frame_dic_list):
        pass

    # def detect_thread(frame, _cuda, _model, _confidence, _num_classes, _nms_thesh, _inp_dim, quadrangle):
    def detect_thread(self, frame, preProcessImg):
        # Do inference with TensorRT
        a = torch.cuda.FloatTensor()

        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        inference_start = time.time()
        self.inputs[0].host = preProcessImg
        trt_outputs = common.do_inference(self.context, bindings=self.bindings, inputs=self.inputs,
                                          outputs=self.outputs, stream=self.stream)
        inference_end = time.time()
        inference_time = inference_end - inference_start

        # print('inference time : %f' % (inference_end - inference_start))

        # Do yolo_layer with pytorch
        write = 0

        strides = [32, 16, 8]
        bbox_attrs = 5 + self.num_classes
        num_anchors = 3

        yolo_start = time.time()
        # for output, shape, anchors in zip(trt_outputs, self.output_shapes, self.yolo_anchors):
        for output, shape, anchors, stride in zip(trt_outputs, self.output_shapes, self.yolo_anchors, strides):
            output = output.reshape(shape)
            trt_output = torch.from_numpy(output).cuda()
            trt_output = trt_output.data

            # trt_output = predict_transform(trt_output, self.inp_dim, anchors, self.num_classes, self.use_cuda)
            # trt_output = predict_transform(trt_output, self.inp_dim, anchors, self.num_classes, self.use_cuda)
            trt_output = predict_transform1(trt_output, stride, anchors, num_anchors, bbox_attrs)
            if type(trt_output) == int:
                continue

            if not write:
                detections = trt_output
                write = 1

            else:
                detections = torch.cat((detections, trt_output), 1)

        nms_start = time.time()
        dets = dynamic_write_results(detections, 0.4, self.num_classes, nms=True, nms_conf=0.4)  # 0.008
        end = time.time() - nms_start
        # print('nms time:  %f' % end)

        return dets
