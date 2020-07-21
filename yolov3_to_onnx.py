#encoding: utf-8
from __future__ import print_function
from collections import OrderedDict
import hashlib
import os.path

# import wget

import onnx      # github网址为https://github.com/onnx/onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

import sys


'''main第二步：解析yolov3.cfg '''
class DarkNetParser(object):
    """定义一个基于DarkNet YOLOv3-608的解析器."""

    def __init__(self, supported_layers):
        """初始化DarkNetParser对象.

        Keyword argument:
        supported_layers -- 一个list，其中每个元素为字符串，表示支持的层，以DarkNet的命名习惯,
        """

        self.layer_configs = OrderedDict()
        self.supported_layers = supported_layers
        self.layer_counter = 0

    def parse_cfg_file(self, cfg_file_path):
        """逐层解析yolov3.cfg文件,以字典形式追加每层的参数到layer_configs

        Keyword argument:
        cfg_file_path -- yolov3.cfg文件的路径
        """

        with open(cfg_file_path, 'rb') as cfg_file:
            remainder = cfg_file.read()
            remainder = remainder.decode('utf-8') # 这一行for py3
            while remainder:
                # 一次次的去处理字符串，如果返回的layer_dict有值，则表示当前已经处理完一个字典
                layer_dict, layer_name, remainder = self._next_layer(remainder)
                if layer_dict:
                    self.layer_configs[layer_name] = layer_dict

        return self.layer_configs

    def _next_layer(self, remainder):
        """将其视为一个字符串，然后以DarkNet的分隔符来逐段处理.
        在最近的分隔符之后，返回层参数和剩下的字符串
        如文件中第一个Conv层 ...

        [convolutional]
        batch_normalize=1
        filters=32
        size=3
        stride=1
        pad=1
        activation=leaky

        ... 会变成如下形式字典:
        {'activation': 'leaky', 'stride': 1, 'pad': 1, 'filters': 32,
        'batch_normalize': 1, 'type': 'convolutional', 'size': 3}.

        '001_convolutional' 是层名layer_name, 后续所有字符以remainder表示的字符串返回

        Keyword argument:
        remainder -- 仍需要处理的字符串
        """

        # head，tail方式
        # 读取'['，然后获取tail
        remainder = remainder.split('[', 1)
        if len(remainder) == 2:
            remainder = remainder[1]
        else:
            return None, None, None
        # 读取‘]’，然后获取tail
        remainder = remainder.split(']', 1)
        if len(remainder) == 2:
            layer_type, remainder = remainder
        else:
            return None, None, None
        # 过滤注释行
        if remainder.replace(' ', '')[0] == '#':
            remainder = remainder.split('\n', 1)[1]

        # 1空行视为分块的分隔符，这里读取head表示的分块
        layer_param_block, remainder = remainder.split('\n\n', 1)

        # 处理得到的分块，并以'\n'将该块划分成行为元素的列表，等待处理
        layer_param_lines = layer_param_block.split('\n')[1:]

        layer_name = str(self.layer_counter).zfill(3) + '_' + layer_type # 当前块命名
        layer_dict = dict(type=layer_type)

        # 如果当前层是支持的，则进行处理，如yolo就不支持
        if layer_type in self.supported_layers:
            for param_line in layer_param_lines:
                if param_line[0] == '#':
                    continue
                # 解析每一行
                param_type, param_value = self._parse_params(param_line)
                layer_dict[param_type] = param_value

        self.layer_counter += 1

        return layer_dict, layer_name, remainder

    def _parse_params(self, param_line):
        """解析每一行参数，当遇到layers时，返回list，其余返回字符串，整数，浮点数类型.

        Keyword argument:
        param_line -- 块中的一行需要解析的参数行
        """
        param_line = param_line.replace(' ', '') # 紧凑一下
        param_type, param_value_raw = param_line.split('=') # 以‘=’划分
        param_value = None

        # 如果当前参数是layers，则以列表形式返回
        if param_type == 'layers':
            layer_indexes = list()
            for index in param_value_raw.split(','):
                layer_indexes.append(int(index))
            param_value = layer_indexes
        # 否则先检测是否是整数，还是浮点数，不然就返回字符串类型
        elif isinstance(param_value_raw, str) and not param_value_raw.isalpha():
            condition_param_value_positive = param_value_raw.isdigit()
            condition_param_value_negative = param_value_raw[0] == '-' and \
                param_value_raw[1:].isdigit()
            if condition_param_value_positive or condition_param_value_negative:
                param_value = int(param_value_raw)
            else:
                param_value = float(param_value_raw)
        else:
            param_value = str(param_value_raw)

        return param_type, param_value


'''main第四步：被第三步类的_make_onnx_node方法调用 '''
class MajorNodeSpecs(object):
    """Helper class用于存储ONNX输出节点的信息，对应DarkNet 层的输出和该层输出通道，
     一些DarkNet层并未被创建，因此没有对应的ONNX 节点，
     不过仍然需要对其进行追踪以建立skip 连接
    """

    def __init__(self, name, channels):
        """ 初始化一个MajorNodeSpecs对象

        Keyword arguments:
        name -- ONNX节点的名称
        channels -- 该节点的输出通道的数量
        """
        self.name = name
        self.channels = channels
        # 对于yolov3.cfg中三层yolo层，这里表示该节点并非onnx节点，默认复制false
        # 其他如卷积，上采样等都是被赋予true
        self.created_onnx_node = False
        if name is not None and isinstance(channels, int) and channels > 0:
            self.created_onnx_node = True


'''main第四步：被第三步类的_make_conv_node方法调用 '''
class ConvParams(object):
    """Helper class用于存储卷积层的超参数,包括在ONNX graph中的前置name和
         为了卷积，偏置，BN等权重期望的维度

    另外该类还扮演着为所有权重生成安全名称的封装，并检查合适的组合搭配
    """

    def __init__(self, node_name, batch_normalize, conv_weight_dims):
        """基于base 节点名称 (e.g. 101_convolutional),BN设置，卷积权重shape的构造器

        Keyword arguments:
        node_name -- YOLO卷积层的base名称
        batch_normalize -- bool值，表示是否使用BN
        conv_weight_dims -- 该层的卷积权重的维度
        """
        self.node_name = node_name
        self.batch_normalize = batch_normalize
        assert len(conv_weight_dims) == 4
        self.conv_weight_dims = conv_weight_dims

    def generate_param_name(self, param_category, suffix):
        """基于两个字符串输入生成一个名称,并检查组合搭配是否合理"""
        assert suffix
        assert param_category in ['bn', 'conv']
        assert(suffix in ['scale', 'mean', 'var', 'weights', 'bias'])
        if param_category == 'bn':
            assert self.batch_normalize
            assert suffix in ['scale', 'bias', 'mean', 'var']
        elif param_category == 'conv':
            assert suffix in ['weights', 'bias']
            if suffix == 'bias':
                assert not self.batch_normalize
        param_name = self.node_name + '_' + param_category + '_' + suffix
        return param_name


'''man第四步：被第三步类的build_onnx_graph方法调用 '''
class WeightLoader(object):
    """Helper class用于载入序列化的权重，
    """

    def __init__(self, weights_file_path):
        """读取YOLOv3权重文件

        Keyword argument:
        weights_file_path --权重文件的路径.
        """
        self.weights_file = self._open_weights_file(weights_file_path)

    def load_conv_weights(self, conv_params):
        """返回权重文件的初始化器和卷积层的输入tensor

        Keyword argument:
        conv_params -- a ConvParams object
        """
        initializer = list()
        inputs = list()
        if conv_params.batch_normalize:
            # 创建BN需要的bias，scale，mean，var等参数
            bias_init, bias_input = self._create_param_tensors(
                conv_params, 'bn', 'bias')
            bn_scale_init, bn_scale_input = self._create_param_tensors(
                conv_params, 'bn', 'scale')
            bn_mean_init, bn_mean_input = self._create_param_tensors(
                conv_params, 'bn', 'mean')
            bn_var_init, bn_var_input = self._create_param_tensors(
                conv_params, 'bn', 'var')
            # 初始化器扩展； 当前层输入的扩展
            initializer.extend(
                [bn_scale_init, bias_init, bn_mean_init, bn_var_init])
            inputs.extend([bn_scale_input, bias_input,
                           bn_mean_input, bn_var_input])
        else:
            # 处理卷积层；  初始化器扩展； 当前层输入的扩展
            bias_init, bias_input = self._create_param_tensors(
                conv_params, 'conv', 'bias')
            initializer.append(bias_init)
            inputs.append(bias_input)

        # 创建卷积层权重；  初始化器扩展； 当前层输入的扩展
        conv_init, conv_input = self._create_param_tensors(
            conv_params, 'conv', 'weights')
        initializer.append(conv_init)
        inputs.append(conv_input)

        return initializer, inputs

    def _open_weights_file(self, weights_file_path):
        """打开Yolov3 DarkNet文件流，并跳过开头.

        Keyword argument:
        weights_file_path -- 权重文件路径
        """
        weights_file = open(weights_file_path, 'rb')
        length_header = 5
        np.ndarray(
            shape=(length_header, ), dtype='int32', buffer=weights_file.read(
                length_header * 4))
        return weights_file

    def _create_param_tensors(self, conv_params, param_category, suffix):
        """用权重文件中，与输入tensors一起的权重去初始化一个初始化器.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        param_name, param_data, param_data_shape = self._load_one_param_type(
            conv_params, param_category, suffix)

        # 调用onnx.helper.make_tensor
        initializer_tensor = helper.make_tensor(
            param_name, TensorProto.FLOAT, param_data_shape, param_data)
        # 调用onnx.helper.make_tensor_value_info
        input_tensor = helper.make_tensor_value_info(
            param_name, TensorProto.FLOAT, param_data_shape)

        return initializer_tensor, input_tensor

    def _load_one_param_type(self, conv_params, param_category, suffix):
        """基于DarkNet顺序进行文件流的反序列化.

        Keyword arguments:
        conv_params -- a ConvParams object
        param_category -- the category of parameters to be created ('bn' or 'conv')
        suffix -- a string determining the sub-type of above param_category (e.g.,
        'weights' or 'bias')
        """
        # 生成合理的名称
        param_name = conv_params.generate_param_name(param_category, suffix)
        channels_out, channels_in, filter_h, filter_w = conv_params.conv_weight_dims

        if param_category == 'bn':
            param_shape = [channels_out]
        elif param_category == 'conv':
            if suffix == 'weights':
                param_shape = [channels_out, channels_in, filter_h, filter_w]
            elif suffix == 'bias':
                param_shape = [channels_out]

        param_size = np.product(np.array(param_shape)) # 计算参数的size
        # 用weights_file.read去逐字节的读取数据并转换
        param_data = np.ndarray(
            shape=param_shape,
            dtype='float32',
            buffer=self.weights_file.read(param_size * 4))
        param_data = param_data.flatten().astype(float)

        return param_name, param_data, param_shape


'''main第三步 '''
class GraphBuilderONNX(object):
    """用于创建ONNX graph的类，基于之前从yolov3.cfg读取的网络结构。该类函数方法有：
        build_onnx_graph : 构建
        _make_onnx_node
        _make_input_tensor
        _get_previous_node_specs
        _make_conv_node
        _make_shortcut_node
        _make_route_node
        _make_upsample_node
    """

    def __init__(self, output_tensors):
        """用所有DarkNet默认参数来初始化；
            然后基于output_tensors指定输出维度；
           以他们的name为key

        Keyword argument:
        output_tensors -- 一个 OrderedDict类型
        """

        self.output_tensors = output_tensors
        self._nodes = list()
        self.graph_def = None
        self.input_tensor = None
        self.epsilon_bn = 1e-5
        self.momentum_bn = 0.99
        self.alpha_lrelu = 0.1
        self.param_dict = OrderedDict()
        self.major_node_specs = list()
        self.batch_size = 1

    def build_onnx_graph(
            self,
            layer_configs,
            weights_file_path,
            verbose=True):
        """基于所有的层配置进行迭代，创建一个ONNX graph，
            然后用下载的yolov3 权重文件进行填充，最后返回该graph定义.

        Keyword arguments:
        layer_configs -- OrderedDict对象，包含所有解析的层的配置
        weights_file_path -- 权重文件的位置
        verbose -- 是否在创建之后显示该graph(default: True)
        """

        for layer_name in layer_configs.keys():

            layer_dict = layer_configs[layer_name]
            # 读取yolov3.cfg中每一层，并将其作为onnx的节点
            major_node_specs = self._make_onnx_node(layer_name, layer_dict)
            # 如果当前为主要节点，则追加起来
            if major_node_specs.name:
                self.major_node_specs.append(major_node_specs)

        outputs = list()
        for tensor_name in self.output_tensors.keys():
            # 将输出节点进行维度扩充
            output_dims = [self.batch_size, ] + \
                self.output_tensors[tensor_name]
            # 调用onnx的helper.make_tensor_value_info构建onnx张量，此时并未填充权重
            output_tensor = helper.make_tensor_value_info(
                tensor_name, TensorProto.FLOAT, output_dims)
            outputs.append(output_tensor)

        inputs = [self.input_tensor]
        weight_loader = WeightLoader(weights_file_path)
        initializer = list()
        # self.param_dict在_make_onnx_node中已处理
        for layer_name in self.param_dict.keys():
            _, layer_type = layer_name.split('_', 1) # 如001_convolutional
            conv_params = self.param_dict[layer_name]
            assert layer_type == 'convolutional'
            initializer_layer, inputs_layer = weight_loader.load_conv_weights(
                conv_params)
            initializer.extend(initializer_layer)
            inputs.extend(inputs_layer)
        del weight_loader

        # 调用onnx的helper.make_graph进行onnx graph的构建
        self.graph_def = helper.make_graph(
            nodes=self._nodes,
            name='YOLOv3-608',
            inputs=inputs,
            outputs=outputs,
            initializer=initializer
        )

        if verbose:
            print(helper.printable_graph(self.graph_def))

        # 调用onnx的helper.make_model进行模型的构建
        model_def = helper.make_model(self.graph_def,
                                      producer_name='NVIDIA TensorRT sample')
        return model_def

    def _make_onnx_node(self, layer_name, layer_dict):
        """输入一个layer参数字典，选择对应的函数来创建ONNX节点，然后将为图创建的重要的信息存储为
           MajorNodeSpec对象

        Keyword arguments:
        layer_name -- layer的名称 (即layer_configs中的key)
        layer_dict -- 一个layer参数字典 (layer_configs的value)
        """
        
        layer_type = layer_dict['type']
        # 先检查self.input_tensor是否为空，为空且第一个块不是net，则报错，否则处理该net
        # 可以看出 这里只在最开始执行一次，因为后续self.input_tensor都不为空。
        if self.input_tensor is None:
            if layer_type == 'net':
                major_node_output_name, major_node_output_channels = self._make_input_tensor(
                    layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name,
                                                  major_node_output_channels)
            else:
                raise ValueError('The first node has to be of type "net".')
        else:
            node_creators = dict()
            node_creators['convolutional'] = self._make_conv_node
            node_creators['shortcut'] = self._make_shortcut_node
            node_creators['route'] = self._make_route_node
            node_creators['upsample'] = self._make_upsample_node

            # 依次处理不同的层，并调用对应node_creators[layer_type]()函数进行处理
            if layer_type in node_creators.keys():
                major_node_output_name, major_node_output_channels = \
                    node_creators[layer_type](layer_name, layer_dict)
                major_node_specs = MajorNodeSpecs(major_node_output_name,
                                                  major_node_output_channels)
            else:
                # 跳过三个yolo层
                print(
                    'Layer of type %s not supported, skipping ONNX node generation.' %
                    layer_type)
                major_node_specs = MajorNodeSpecs(layer_name,
                                                  None)
        return major_node_specs

    def _make_input_tensor(self, layer_name, layer_dict):
        """为net layer创建输入tensor，并存储对应batch size.可以看出，该函数只被调用一次

        Keyword arguments:
        layer_name -- 层的名字 (如 layer_configs中key)
        layer_dict -- 一个layer参数字典( layer_configs中的value)
        """
        batch_size = layer_dict['batch']
        channels = layer_dict['channels']
        height = layer_dict['height']
        width = layer_dict['width']
        self.batch_size = batch_size

        # 用onnx.helper.make_tensor_value_info构建onnx张量节点
        input_tensor = helper.make_tensor_value_info(
            str(layer_name), TensorProto.FLOAT, [
                batch_size, channels, height, width])
        self.input_tensor = input_tensor

        return layer_name, channels

    def _get_previous_node_specs(self, target_index=-1):
        """获取之前创建好的onnx节点(跳过那些没生成的节点，比如yolo节点).
        target_index可以能够直接跳到对应节点.

        Keyword arguments:
        target_index -- 可选的参数，帮助跳到具体索引(default: -1 表示跳到前一个元素)
        """

        # 通过反向遍历，找到最后一个（这里是第一个）created_onnx_node为真的节点
        previous_node = None
        for node in self.major_node_specs[target_index::-1]:
            if node.created_onnx_node:
                previous_node = node
                break
        assert previous_node is not None
        return previous_node

    def _make_conv_node(self, layer_name, layer_dict):
        """用可选的bn和激活函数nonde去创建一个onnx的卷积node

        Keyword arguments:
        layer_name -- 层的名字 (如 layer_configs中key)
        layer_dict -- 一个layer参数字典( layer_configs中的value)
        """
        # 先找最近的一个节点
        previous_node_specs = self._get_previous_node_specs()

        ''' i) 处理卷积层'''
        # 构建该层的inputs，通道等等信息
        inputs = [previous_node_specs.name]
        previous_channels = previous_node_specs.channels
        kernel_size = layer_dict['size']
        stride = layer_dict['stride']
        filters = layer_dict['filters']
        # 检测该层是否有bn
        batch_normalize = False
        if 'batch_normalize' in layer_dict.keys(
        ) and layer_dict['batch_normalize'] == 1:
            batch_normalize = True

        kernel_shape = [kernel_size, kernel_size]
        weights_shape = [filters, previous_channels] + kernel_shape
        # 构建卷积层的参数层的实例
        conv_params = ConvParams(layer_name, batch_normalize, weights_shape)

        strides = [stride, stride]
        dilations = [1, 1]
        # 调用ConvParams.generate_param_name生成合适的参数名称
        weights_name = conv_params.generate_param_name('conv', 'weights')
        inputs.append(weights_name)
        if not batch_normalize:
            bias_name = conv_params.generate_param_name('conv', 'bias')
            inputs.append(bias_name)

        # 用onnx.helper.make_node构建onnx的卷积节点
        conv_node = helper.make_node(
            'Conv',
            inputs=inputs,
            outputs=[layer_name],
            kernel_shape=kernel_shape,
            strides=strides,
            auto_pad='SAME_LOWER',
            dilations=dilations,
            name=layer_name
        )
        self._nodes.append(conv_node)

        inputs = [layer_name]
        layer_name_output = layer_name

        ''' ii) 处理BN层'''
        if batch_normalize:
            layer_name_bn = layer_name + '_bn'
            bn_param_suffixes = ['scale', 'bias', 'mean', 'var']
            for suffix in bn_param_suffixes:
                bn_param_name = conv_params.generate_param_name('bn', suffix)
                inputs.append(bn_param_name)
            batchnorm_node = helper.make_node(
                'BatchNormalization',
                inputs=inputs,
                outputs=[layer_name_bn],
                epsilon=self.epsilon_bn,
                momentum=self.momentum_bn,
                name=layer_name_bn
            )
            self._nodes.append(batchnorm_node)

            inputs = [layer_name_bn]
            layer_name_output = layer_name_bn

        ''' iii) 处理激活函数'''
        if layer_dict['activation'] == 'leaky':
            layer_name_lrelu = layer_name + '_lrelu'

            lrelu_node = helper.make_node(
                'LeakyRelu',
                inputs=inputs,
                outputs=[layer_name_lrelu],
                name=layer_name_lrelu,
                alpha=self.alpha_lrelu
            )
            self._nodes.append(lrelu_node)
            inputs = [layer_name_lrelu]
            layer_name_output = layer_name_lrelu
        elif layer_dict['activation'] == 'linear':
            pass
        else:
            print('Activation not supported.')

        self.param_dict[layer_name] = conv_params
        return layer_name_output, filters

    def _make_shortcut_node(self, layer_name, layer_dict):
        """从DarkNet graph中读取信息，基于onnx 的add 节点创建shortcut 节点.

        Keyword arguments:
        layer_name -- 层的名字 (如 layer_configs中key)
        layer_dict -- 一个layer参数字典( layer_configs中的value)
        """
        shortcut_index = layer_dict['from'] # 当前层与前面哪层shorcut
        activation = layer_dict['activation']
        assert activation == 'linear'

        first_node_specs = self._get_previous_node_specs() # 最近一层
        second_node_specs = self._get_previous_node_specs(
            target_index=shortcut_index) # 前面具体需要shorcut的层
        assert first_node_specs.channels == second_node_specs.channels
        channels = first_node_specs.channels
        inputs = [first_node_specs.name, second_node_specs.name]
        # 用onnx.helper.make_node创建节点
        shortcut_node = helper.make_node(
            'Add',
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(shortcut_node)
        return layer_name, channels

    def _make_route_node(self, layer_name, layer_dict):
        """如果来自DarkNet配置的layer参数只有一个所以，那么接着在指定（负）索引上创建节点
           否则，创建一个onnx concat 节点来实现路由特性.

        Keyword arguments:
        layer_name -- 层的名字 (如 layer_configs中key)
        layer_dict -- 一个layer参数字典( layer_configs中的value)
        """
        # 处理yolov3.cfg中[route]
        route_node_indexes = layer_dict['layers']

        if len(route_node_indexes) == 1:
            split_index = route_node_indexes[0]
            assert split_index < 0
            # Increment by one because we skipped the YOLO layer:
            split_index += 1
            self.major_node_specs = self.major_node_specs[:split_index]
            layer_name = None
            channels = None
        else:
            inputs = list()
            channels = 0
            for index in route_node_indexes:
                if index > 0:
                    # Increment by one because we count the input as a node (DarkNet
                    # does not)
                    index += 1
                route_node_specs = self._get_previous_node_specs(
                    target_index=index)
                inputs.append(route_node_specs.name)
                channels += route_node_specs.channels
            assert inputs
            assert channels > 0

            route_node = helper.make_node(
                'Concat',
                axis=1,
                inputs=inputs,
                outputs=[layer_name],
                name=layer_name,
            )
            self._nodes.append(route_node)
        return layer_name, channels

    def _make_upsample_node(self, layer_name, layer_dict):
        """创建一个onnx的Upsample节点.

        Keyword arguments:
        layer_name -- 层的名字 (如 layer_configs中key)
        layer_dict -- 一个layer参数字典( layer_configs中的value)
        """
        upsample_factor = float(layer_dict['stride'])
        previous_node_specs = self._get_previous_node_specs()
        inputs = [previous_node_specs.name]
        channels = previous_node_specs.channels
        assert channels > 0
        upsample_node = helper.make_node(
            'Upsample',
            mode='nearest',
            # For ONNX versions <0.7.0, Upsample nodes accept different parameters than 'scales':
            scales=[1.0, 1.0, upsample_factor, upsample_factor],
            inputs=inputs,
            outputs=[layer_name],
            name=layer_name,
        )
        self._nodes.append(upsample_node)
        return layer_name, channels


def generate_md5_checksum(local_path):
    """计算本地文件的md5

    Keyword argument:
    local_path -- 本地文件路径
    """
    with open(local_path) as local_file:
        data = local_file.read()
        return hashlib.md5(data).hexdigest()


def download_file(local_path, link, checksum_reference=None):
    """下载指定url到本地，并进行摘要校对.

    Keyword arguments:
    local_path -- 本地文件存储路径
    link -- 需要下载的url
    checksum_reference -- expected MD5 checksum of the file
    """
    # if not os.path.exists(local_path):
    #     print('Downloading from %s, this may take a while...' % link)
    #     wget.download(link, local_path)
    #     print()

    if checksum_reference is not None:
        checksum = generate_md5_checksum(local_path)
        if checksum != checksum_reference:
            raise ValueError(
                'The MD5 checksum of local file %s differs from %s, please manually remove \
                 the file and try again.' %
                (local_path, checksum_reference))

    return local_path


def main():

    """Run the DarkNet-to-ONNX conversion for YOLOv3-608."""

    # 注释掉下面的部分，
#    if sys.version_info[0] > 2:
#        raise Exception("This is script is only compatible with python2, please re-run this script \
#    with python2. The rest of this sample can be run with either version of python")

    ''' 1 - 下载yolov3的配置文件，并进行摘要验证'''
#    cfg_file_path = download_file(
#        'yolov3.cfg',
#  'https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg',
#        'b969a43a848bbf26901643b833cfb96c')

    #cfg_file_path = "yolov3-608.cfg"
    cfg_file_path = "cfg/yolov3-416.cfg"

    # DarkNetParser将会只提取这些层的参数，类型为'yolo'的这三层不能很好的解析，
    # 因为他们包含在后续的后处理中；
    supported_layers = ['net', 'convolutional', 'shortcut', 'route', 'upsample']

    ''' 2 - 创建一个DarkNetParser对象，并生成一个OrderedDict，包含cfg文件读取的所有层配置'''
    parser = DarkNetParser(supported_layers)
    layer_configs = parser.parse_cfg_file(cfg_file_path)
    # 在解析完之后，不再需要该对象
    del parser

    ''' 3 - 实例化一个GraphBuilderONNX类对象，用已知输出tensor维度进行初始化'''
    # 在上面的layer_config，有三个输出是需要知道的，CHW格式
    output_tensor_dims = OrderedDict()
    #output_tensor_dims['082_convolutional'] = [255, 19, 19]
    #output_tensor_dims['094_convolutional'] = [255, 38, 38]
    #output_tensor_dims['106_convolutional'] = [255, 76, 76]
    output_tensor_dims['082_convolutional'] = [255, 13, 13]
    output_tensor_dims['094_convolutional'] = [255, 26, 26]
    output_tensor_dims['106_convolutional'] = [255, 52, 52]

    # 内置yolov3的一些默认参数来进行实例化
    builder = GraphBuilderONNX(output_tensor_dims)

    ''' 4 - 调用GraphBuilderONNX的build_onnx_graph方法
           用之前解析好的层配置信息和权重文件，生成ONNX graph'''
    ''' 从作者官网下载yolov3的权重文件，以此填充tensorrt的network '''
    """weights_file_path = download_file(
        'yolov3.weights',
        'https://pjreddie.com/media/files/yolov3.weights',
        'c84e5b99d0e52cd466ae710cadf6d84c')
    """

    #weights_file_path = "yolov3-608.weights"
    weights_file_path = "weights/yolov3.weights"

    
    yolov3_model_def = builder.build_onnx_graph(
        layer_configs=layer_configs,
        weights_file_path=weights_file_path,
        verbose=True)
    # 模型定义结束，删除builder对象
    del builder

    ''' 5 - 在ONNX模型定义上进行健全检查'''
    onnx.checker.check_model(yolov3_model_def)

    ''' 6 - 序列化生成的ONNX graph到文件'''
    #output_file_path = 'yolov3.onnx'
    output_file_path = 'weights/yolov3.onnx'
    onnx.save(yolov3_model_def, output_file_path)

if __name__ == '__main__':
    main()


