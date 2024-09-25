# Copyright (c) 2020-2023, AiM Future Inc.
#
# This program or software including the accompanying associated documentation ("Software") is the proprietary software of AiM Future Inc. and or its licensors, and may only be used, duplicated, modified or distributed pursuant to the terms and conditions of a separate written license agreement between you and AiM Future Inc. ("Authorized License"). Except as set forth in an Authorized License, AiM Future Inc. grants no license (express or implied), rights to use, or waiver of any kind with respect to the Software, and AiM Future Inc. expressly reserves all rights in and to the Software and all intellectual property therein. If you have no Authorized License, then you have no rights to use the Software in any ways, and should immediately notify AiM Future Inc. and discontinue all use of the Software.
#
#

import tensorflow as tf
import keras
import torch
import torch.nn as nn
import numpy as np
import re
import sys
import os
import tempfile
from collections import OrderedDict
#import synabro


class TorchManager:

    def __init__(self,
                 model,  # model
                 input_shape,  # input shape
                 model_name=None,  # model name
                 input_names=None,  # input tensor name
                 output_names=None,  # output tensor names
                 probe_names=None,  # probed tensor names
                 check_trace=True
                 ):
        f"""    Initialize an {__class__.__name__} object.

    [Arguments]
        Name        Required/Default    Description
        ----------------------------------------------------------------------------------------
        models         Yes              Torch model object
        input_shape    Yes              tensor input shape
        model_name     No     None      model name
        input_names    No     None      forced model inputs,     looked up from Torch graph
        output_names   No     None      forced model outputs,    looked up from Torch graph
        probe_names    No     None      probed internal tensors, looked up from Torch graph

    ============================================================================================
      How to build internal data structures from a Torch graph.
    ============================================================================================

    As 'graph' is already flattened, only Torch native operators are in the graph. In analyzing Torch graph, you should be familiar with 'torch.Node', 'torch.Value', and there relation.

    A torch.Graph object maintains information as a graph of torch.Node objects. While a torch.Node object is similar to the Keras model layers, it only keeps the symbolic information to build the graph and all concrete information should be obtained by inspecting its coupled torch.Value objects.

    For better understanding, with negligible conceptual errors, you may think as described below.

        objects                 Description
        ----------------------------------------------------------------------------------------
        xn                      torch.Node  object
          xn.inputs()           torch.Value objects for inputs, hyperparameters
          xn.outputs()          torch.Value objects for outputs
          xn.kind()             Torch C++ operator name
          xn.schema()           Torch C++ operator schema

        xv                      torch.Value object
          xv.node()             producer/predecessor torch.Node
          xv.uses().user()      consumer/successor   torch.Node objects

    Relation between torch.Node objects and their associated torch.Value objects is depicted below.

              torch.Node objects connected by iterator.next() in execution order 
             -------------------------------------------------------------------->

        +-----------+   iterator.next()   +-----------+   iterator.next()   +-----------+
        |  node_0   |  ------ ... ----->  |  node_1   |  ------ ... ----->  |  node_2   |
        +-----------+                     +-----------+                     +-----------+
              :                                 :                                 :
              : outputs()              inputs() : outputs()              inputs() :
              : --------> +---------+ <-------- : --------> +---------+ <-------- :
              : <-------- | value_A | --------> : <-------- | value_B | --------> :
              :    node() +---------+  uses()   :    node() +---------+  uses()   :

    Note that 'xn.inputs()' may include input tensors, kernel/bias tensors, and hyperparameters. Although all torch.Node inputs are represented as torch.Value objects, they should be processed in a totally different manner, which is described below.

        torch.Value objects     Description
        ----------------------------------------------------------------------------------------
        input tensors           tensor   stored in SIGNALS[xv]["tensor"]
        kernel/bias tensors     loaded from 'model.state_dict()', stored with output tensors
        hyperparameters         constant stored in SIGNALS[xv]["const"]

    'SIGNALS[xv]["type"]' indicates if 'SIGNALS[xv]' is a tensor or a constant.
    """
        if (len(input_shape) == 3):   input_shape = (1,) + tuple(input_shape)

        model.eval()

        params = model.state_dict()

        inputs = torch.zeros(input_shape, dtype=torch.float32)
        scriptmodule = torch.jit.trace(model, inputs, check_trace=check_trace)
        graph = scriptmodule.inlined_graph

        ##  Initialize dictionaries to record torch.Value objects.
        graph_tensors = {}  # record all torch.Value objects keyed by name
        SIGNALS = {}  # record all signals to build the execution plan

        _xniter = graph.nodes()  # torch.Graph only accessible by Python iterator
        while (True):
            try:
                xn = next(_xniter)
                for i in range(xn.outputsSize()):
                    xv = xn.outputsAt(i)
                    SIGNALS[xv] = {  # initialize 'SIGNALS' dictionary empty
                        "type": None,  # "type"   indicates one of ["tensor","const"]
                        "tensor": None,  # "tensor" stores KerasTensor object
                        "const": None  # "const"  stores numeric constant
                    }
                    graph_tensors[xv.debugName()] = xv
            except StopIteration:
                break

        ##  Initialize a dictionary to record hyperparameters: <Round-1>
        ##
        ##  As some hyperparameters can only be recorded by inspecting input/output tensors, which may
        ##  be dynamic, only hyperparameters obtained in the static context are recored here.
        _xniter = graph.nodes()
        while (True):
            try:
                xn = next(_xniter)
                kind = xn.kind()

                if (kind == "prim::Constant"):  # constant
                    yv = xn.output()
                    SIGNALS[yv].update({
                        "type": "const",
                        "const": yv.toIValue()
                    })
                elif (kind in ["prim::ListConstruct", "prim::TupleConstruct"]):
                    # list of constants recorded in the static context
                    yv = xn.output()  # updates can be made when all inputs are "const".
                    xvs = [xn.inputsAt(_) for _ in range(xn.inputsSize())]
                    if all(["const" == _ for _ in [SIGNALS[_]["type"] for _ in xvs]]):
                        SIGNALS[yv].update({
                            "type": "const",
                            "const": [SIGNALS[_]["const"] for _ in xvs]
                        })
            except StopIteration:
                break

        ##  Record the model inputs and outputs.
        graph_inputs = []  # torch.Value objects for model inputs
        graph_outputs = []  # torch.Value objects for model outputs

        if (input_names):  # forced model inputs
            graph_inputs += [graph_tensors[_] for _ in input_names]
            print(f"[I] Input tensors are forced to be {input_names}.")
        else:  # original model inputs
            _xviter = graph.inputs()  # torch.Graph only accessible by Python iterator
            while (True):
                try:
                    xv = next(_xviter)
                    if (str(xv.type()) == "Tensor"):
                        SIGNALS[xv] = {  # original model inputs should be recorded explicitly.
                            "type": None,
                            "tensor": None,
                            "const": None
                        }
                        graph_inputs.append(xv)
                except StopIteration:
                    break

        assert len(graph_inputs) == 1, "Unable to handle multiple model inputs"

        if (output_names):  # forced model outputs
            graph_outputs += [graph_tensors[_] for _ in output_names]
            print(f"[I] Output tensors are forced to be {output_names}.")
        else:  # original model outputs
            _xviter = graph.outputs()  # torch.Graph only accessible by Python iterator
            while (True):
                try:
                    xv = next(_xviter)
                    if ("Tensor" in str(xv.type())):  graph_outputs.append(xv)
                except StopIteration:
                    break

        ##  Record the interal signals to probe for debugging.
        ##
        ##  Probed signals are added to the model outputs, which can be easily inspected from outside.
        ##  Probed signals should be indicated by their name in torch.Graph, precisely inlined_graph.
        if (probe_names):
            for probe_name in probe_names:
                xv = graph_tensors[probe_name]
                graph_outputs.append(xv)
                graph.registerOutput(xv)
            print(f"[I] Probed tensors are set to be {probe_names}.")

        ##  Generate the keras model.
        self.model = model
        self.model_name = model_name
        self.scriptmodule = scriptmodule
        self.input_shape = input_shape
        self.input_names = input_names
        self.output_names = output_names
        self.probe_names = probe_names
        self.graph = graph
        self.graph_inputs = graph_inputs
        self.graph_outputs = graph_outputs
        self.params = params
        self.SIGNALS = SIGNALS

    def generate_keras(self,
                       summary_file=None,  # output Keras model summary
                       keras_file=None,  # output Keras model
                       tflite_file=None,  # output TFLite model
                       log_file=None,  # log file
                       output_dir="."  # output directory
                       ):
        f"""    Generate a Keras model from the Torch graph.

    [Arguments]
        Name        Required/Default    Description
        --------------------------------------------------------------------------------------
        summary_file   No     None      if provided, output Keras summary file
        keras_file     No     None      if provided, output Keras model file
        tflite_file    No     None      if provided, output TensorFlow Lite model file
        log_file       No     None      if provided, output logs redirected to 'log_file'
        output_dir     No     "."       all files are generated in 'output_dir'

    [Outputs]
        keras_model                     output keras.Model object
    """
        if not (os.path.exists(output_dir)): os.system(f"mkdir -p {output_dir}")

        ##  Create a Keras model from the Torch graph.
        keras_model = self._convert_torch2keras(log_file, output_dir)

        ##  Apply custom optimization to the Keras model.
        if (self.model_name.startswith("yolox")):
            mosaic = (self.model_name != "yolox_darknet")
            keras_model = __class__._optimize_yolox(keras_model, mosaic=mosaic)
        elif (self.model_name == "nanodet"):
            keras_model = __class__._optimize_nanodet(keras_model)

        ##  Export the Keras model.
        """
        synabro.KerasManager.export(
            keras_model,
            summary_file=summary_file,
            keras_file=keras_file,
            tflite_file=tflite_file,
            output_dir=output_dir
        )
        """
        return keras_model




    def dump_graph(self, dump_file=None, output_dir="."):
        f"""    Generate a Keras model from the Torch graph.

    [Arguments]
        Name        Required/Default    Description
        --------------------------------------------------------------------------------------
        dump_file      No     None      if provided, graph dump redirected to 'dump_file'
        output_dir     No     "."       all files are generated in 'output_dir'
    """
        if not (os.path.exists(output_dir)): os.system(f"mkdir -p {output_dir}")

        if (dump_file):
            open(f"{output_dir}/{dump_file}", "w").write(str(self.graph))
        else:
            print(self.graph)

    def _convert_torch2keras(self, log_file, output_dir):
        params = self.params
        SIGNALS = self.SIGNALS
        if (log_file):
            print(f"[I] Torch-to-Keras conversion logs are written to '{output_dir}/{log_file}'.")
            sys.stdout = open(f"{output_dir}/{log_file}", "w")

        ##  Traverse the graph until reaching the model input.
        if (self.input_names):
            _xniter = self.graph.nodes()
            while (True):
                try:
                    xn = next(_xniter)
                    if (xn == self.graph_inputs[0].node()): break
                except StopIteration:
                    break

            yv = xn.output()
            found = re.findall(fr"^%{yv.debugName()} : .*\((.*), strides", str(xn))[0]
            input_shape = [int(_) for _ in found.split(",")]
        else:
            yv = self.graph_inputs[0]
            input_shape = self.input_shape
            _xniter = self.graph.nodes()  # iterator pointing to the beginning

        INPUTS = keras.layers.Input(
            np.array(input_shape)[[2, 3, 1]], name=yv.debugName().replace(".", "_")
        )
        SIGNALS[yv].update({"type": "tensor", "tensor": INPUTS})
        print(f"[I] Convert torch Graph operators to TensorFlow/Keras layers.\n")
        print("  {:24}{:20}{}".format("Layer", "Output", "Shape"))
        print("  " + "-" * 68)
        print("  {:24}{:20}{}".format("Input", yv.debugName(), INPUTS.shape))

        ##  Process hidden layers.
        output_nodes = [_.node() for _ in self.graph_outputs]
        while (len(output_nodes) > 0):
            try:
                xn = next(_xniter)
                kind = xn.kind()
                xvs = [xn.inputsAt(_) for _ in range(xn.inputsSize())]
                yvs = [xn.outputsAt(_) for _ in range(xn.outputsSize())]

                if (kind == "aten::_convolution"):  # [Conv2D], [DepthwiseConv2D]
                    xt = SIGNALS[xvs[0]]["tensor"]  # input tensor
                    stride = SIGNALS[xvs[3]]["const"]  # stride
                    _padding = SIGNALS[xvs[4]]["const"]  # input padding
                    dilation = SIGNALS[xvs[5]]["const"]
                    transposed = SIGNALS[xvs[6]]["const"]
                    _o_padding = SIGNALS[xvs[7]]["const"]
                    groups = SIGNALS[xvs[8]]["const"]  # number of groups

                    scope = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    print(scope, xn.scopeName())

                    k = params[f"{scope}.weight"]  # kernel tensor
                    YC, XC, KH, KW = k.size()
                    depthwise = (groups > 1 and groups == YC)
                    b = params.get(f"{scope}.bias")  # bias tensor
                    use_bias = (b != None)

                    if (stride == [2, 2] or (stride == [1, 1] and _padding != [KH // 2, KW // 2])):
                        xt = keras.layers.ZeroPadding2D(padding=_padding)(xt)
                        padding = "valid"
                    else:
                        padding = "same"

                    kwargs = {
                        "kernel_size": (KH, KW),
                        "strides": stride,
                        "padding": padding,
                        "use_bias": use_bias
                    }

                    if (depthwise):
                        k = k.permute(2, 3, 0, 1)  # NCHW to HWCN, where XC=1 in Torch DepthWiseConv2D
                        kwargs.update({
                            "weights": [k, b] if (use_bias) else [k]
                        })
                        yt = keras.layers.DepthwiseConv2D(**kwargs)(xt)
                        print("  {:24}{:20}{}".format("DepthwiseConv2D", yvs[0].debugName(), yt.shape))
                    else:
                        k = k.permute(2, 3, 1, 0)  # NCHW to HWCN
                        kwargs.update({
                            "filters": YC,
                            "weights": [k, b] if (use_bias) else [k]
                        })
                        yt = keras.layers.Conv2D(**kwargs)(xt)
                        print("  {:24}{:20}{}".format("Conv2D", yvs[0].debugName(), yt.shape))

                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                elif (kind == "aten::max_pool2d"):  # [MaxPooling2D]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    KH, KW = SIGNALS[xvs[1]]["const"]
                    stride = SIGNALS[xvs[2]]["const"]
                    _padding = SIGNALS[xvs[3]]["const"]

                    if (_padding not in [[0, 0], [KH // 2, KW // 2]]):
                        xt = tf.pad(xt,
                                    [[0, 0], [KH // 2, KH // 2], [KW // 2, KW // 2], [0, 0]],
                                    # 4D paddings for [N,H,W,C]
                                    constant_values=-float("inf")
                                    )
                        padding = "valid"
                    else:
                        padding = "same"

                    yt = keras.layers.MaxPooling2D((KH, KW), stride, padding)(xt)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("MaxPooling2D", yvs[0].debugName(), yt.shape))
                elif (kind == "aten::adaptive_avg_pool2d"):  # [GlobalAveragePooling2D]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    output_size = SIGNALS[xvs[1]]["const"]

                    if (output_size == [1, 1]):
                        yt = keras.layers.GlobalAveragePooling2D(keepdims=True)(xt)
                    else:
                        raise NotImplementedError(f"Unable to handle output_size='{output_size}'")

                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("MaxPooling2D", yvs[0].debugName(), yt.shape))

                elif (kind == "aten::batch_norm"):  # [BatchNormalization]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    epsilon = SIGNALS[xvs[7]]["const"]
                    scope = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    yt = keras.layers.BatchNormalization(
                        epsilon=epsilon,
                        weights=[
                            params[f"{scope}.weight"],
                            params[f"{scope}.bias"],
                            params[f"{scope}.running_mean"],
                            params[f"{scope}.running_var"]
                        ]
                    )(xt)
                    print("  {:24}{:20}{}".format("BatchNormalization", yvs[0].debugName(), yt.shape))
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})

                elif (kind in ["aten::view", "aten::reshape"]):  # [TF.Reshape]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    _size = SIGNALS[xvs[1]]["const"]
                    ndims = len(_size)

                    if (-1 in _size):  # impossible to auto-calculate by -1 in TensorFlow
                        _dim = _size.index(-1)
                        _size[_dim] = -int(np.prod(xt.shape[1:]) / np.prod(_size[1:]))

                    if (ndims == 5):
                        shape = [-1, _size[3], _size[4], _size[1], _size[2]]
                    elif (ndims == 4):
                        shape = [-1, _size[2], _size[3], _size[1]]
                    elif (ndims == 3):
                        shape = [-1, _size[2], _size[1]]
                    else:
                        raise NotImplementedError

                    yt = tf.reshape(xt, shape)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                elif (kind == "aten::flatten"):  # [TF.Reshape]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    _dim0 = SIGNALS[xvs[1]]["const"]
                    _dim1 = SIGNALS[xvs[2]]["const"]
                    ndims = len(xt.shape)

                    if (ndims == 4):
                        dim0, dim1 = np.array([0, 3, 1, 2])[[_dim0, _dim1]]
                        shape = [xt.shape[_] for _ in range(dim0)] \
                                + [np.prod(xt.shape[dim0:dim1 + 1])] \
                                + [xt.shape[_] for _ in range(dim1 + 1, 4)]
                        shape = [-1 if (_ == None) else _ for _ in shape]
                    else:
                        raise NotImplementedError

                    yt = tf.reshape(xt, shape)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                elif (kind == "aten::contiguous"):  # [TF.Reshape]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": xt})
                elif (kind in ["aten::transpose", "aten::permute"]):  # [TF.Transpose]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    ndims = len(xt.shape)

                    if (kind == "aten::transpose"):
                        _dim0 = SIGNALS[xvs[1]]["const"]
                        _dim1 = SIGNALS[xvs[2]]["const"]

                        if (ndims == 5):
                            dim0, dim1 = np.array([0, 3, 4, 1, 2])[[_dim0, _dim1]]
                        elif (ndims == 4):
                            dim0, dim1 = np.array([0, 3, 1, 2])[[_dim0, _dim1]]
                        else:
                            raise NotImplementedError

                        perm = list(range(ndims))
                        perm[dim0] = dim1
                        perm[dim1] = dim0
                    else:
                        _dims = SIGNALS[xvs[1]]["const"]

                        if (ndims == 3):
                            perm = list(np.array([0, 2, 1])[_dims])
                        else:
                            raise NotImplementedError(f"ndims={ndims}, _dims={_dims}")

                    yt = tf.transpose(xt, perm) if (sorted(perm) != perm) else xt
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                elif (kind == "aten::split_with_sizes"):  # [TF.Split]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    split_sizes = SIGNALS[xvs[1]]["const"]
                    _dim = SIGNALS[xvs[2]]["const"]
                    ndims = len(xt.shape)

                    if (ndims == 4):
                        axis = [0, 3, 1, 2][_dim]
                    elif (ndims == 3):
                        axis = [0, 2, 1][_dim]

                    yt = tf.split(xt, split_sizes, axis)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                elif (kind == "aten::chunk"):  # [TF.Split]
                    xt = SIGNALS[xvs[0]]["tensor"]
                    chunks = SIGNALS[xvs[1]]["const"]
                    _dim = SIGNALS[xvs[2]]["const"]

                    if (ndims == 5):
                        axis = [0, 3, 4, 1, 2][_dim]
                    elif (ndims == 4):
                        axis = [0, 3, 1, 2][_dim]
                    else:
                        raise NotImplementedError

                    yts = tf.split(xt, chunks, axis)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yts})

                elif (kind in ["aten::upsample_nearest2d", "aten::upsample_bilinear2d"]):  # [UpSampling2D]
                    xt = SIGNALS[xvs[0]]["tensor"]

                    if (kind == "aten::upsample_nearest2d"):
                        output_size = SIGNALS[xvs[1]]["const"]
                        scale_factors = SIGNALS[xvs[2]]["const"]
                    else:
                        output_size = SIGNALS[xvs[1]]["const"]
                        align_corners = SIGNALS[xvs[2]]["const"]
                        scale_factors = SIGNALS[xvs[3]]["const"]

                    if (scale_factors == None):
                        scale_factors = np.array(output_size) // xt.shape[1:3]

                    yt = keras.layers.UpSampling2D(scale_factors)(xt)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("UpSampling2D", yvs[0].debugName(), yt.shape))
                elif (kind == "aten::cat"):  # [Concatenate]
                    xts = SIGNALS[xvs[0]]["tensor"]
                    _dim = SIGNALS[xvs[1]]["const"]
                    ndims = len(xts[0].shape)

                    if (ndims == 4):
                        axis = [0, 3, 1, 2][_dim]
                    elif (ndims == 3):
                        axis = [0, 2, 1][_dim]
                    else:
                        raise NotImplementedError(f"Unknown shape='{xts[0].shape}' provided")

                    yt = keras.layers.Concatenate(axis)(xts)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("Concatenate", yvs[0].debugName(), yt.shape))
                elif (kind in ["aten::add", "aten::add_"]):  # [Add]
                    xt0_type = SIGNALS[xvs[0]]["type"]
                    xt1_type = SIGNALS[xvs[1]]["type"]
                    alpha = SIGNALS[xvs[2]]["const"]
                    assert alpha == 1, f"Unable to handle non-1 alpha='{alpha}'"

                    if (xt0_type == "tensor" and xt1_type == "tensor"):
                        yt = keras.layers.Add()([SIGNALS[xvs[0]]["tensor"], SIGNALS[xvs[1]]["tensor"]])
                        SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                        print("  {:24}{:20}{}".format("Add", yvs[0].debugName(), yt.shape))
                    elif (xt0_type == "tensor" or xt1_type == "tensor"):
                        yt = xt0 + xt1
                        SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                        print("  {:24}{:20}{}".format("Add", yvs[0].debugName(), yt.shape))
                    else:
                        yt = xt0 + xt1
                        SIGNALS[yvs[0]].update({"type": "const", "const": yt})
                elif (kind == "aten::mul"):  # [Multiply]
                    if all(["tensor" == _ for _ in [SIGNALS[_]["type"] for _ in xvs]]):
                        xts = [SIGNALS[_]["tensor"] for _ in xvs]
                        yt = keras.layers.Multiply()(xts)
                        SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    elif all(["const" == _ for _ in [SIGNALS[_]["type"] for _ in xvs]]):
                        yt = np.prod([SIGNALS[_]["const"] for _ in xvs])
                        SIGNALS[yvs[0]].update({"type": "const", "const": yt})
                    else:
                        raise NotImplementedError
                    print("  {:24}{:20}{}".format("Multiply", yvs[0].debugName(), yt.shape))

                elif (kind in ["aten::relu_",  # [Activation]
                               "aten::sigmoid", "aten::hardsigmoid",
                               "aten::silu", "aten::silu_"]):
                    if (kind == "aten::relu_"):
                        activation = keras.activations.relu
                    elif (kind == "aten::sigmoid"):
                        activation = keras.activations.sigmoid
                    elif (kind == "aten::hardsigmoid"):
                        activation = keras.activations.hard_sigmoid
                    else:
                        activation = keras.activations.swish

                    xt = SIGNALS[xvs[0]]["tensor"]
                    yt = activation(xt)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("Sigmoid", yvs[0].debugName(), yt.shape))
                elif (kind == "aten::hardswish_"):  # [Activation] HardSwish
                    xt0 = SIGNALS[xn.input()]["tensor"]
                    xt1 = keras.activations.hard_sigmoid(xt0)
                    yt = keras.layers.Multiply()([xt0, xt1])
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("Activation", yvs[0].debugName(), yt.shape))
                elif (kind == "aten::leaky_relu_"):  # [Activation] LeakyReLU
                    xt = SIGNALS[xvs[0]]["tensor"]
                    alpha = SIGNALS[xvs[1]]["const"]
                    yt = keras.layers.LeakyReLU(alpha)(xt)
                    SIGNALS[yvs[0]].update({"type": "tensor", "tensor": yt})
                    print("  {:24}{:20}{}".format("Activation", yvs[0].debugName(), yt.shape))

                elif (kind in ["prim::ListConstruct", "prim::TupleConstruct"]):
                    if ("tensor" in [SIGNALS[_]["type"] for _ in xvs]):
                        SIGNALS[yvs[0]].update({
                            "type": "tensor",
                            "tensor": [SIGNALS[_]["tensor"] for _ in xvs]
                        })
                    elif all(["const" == _ for _ in [SIGNALS[_]["type"] for _ in xvs]]):
                        SIGNALS[xn.output()].update({
                            "type": "const",
                            "const": [SIGNALS[_]["const"] for _ in xvs]
                        })
                elif (kind in ["prim::ListUnpack", "prim::TupleUnpack"]):
                    _type = SIGNALS[xvs[0]]["type"]
                    xts0 = SIGNALS[xvs[0]][_type]

                    for i in range(len(yvs)):
                        SIGNALS[yvs[i]].update({"type": _type, _type: xts0[i]})
                elif (kind == "aten::size"):
                    xt = SIGNALS[xvs[0]]["tensor"]
                    _axis = SIGNALS[xvs[1]]["const"]

                    if (len(xt.shape) == 4):
                        axis = {-1: 2, 0: 0, 1: 3, 2: 1, 3: 2}[_axis]
                    else:
                        raise NotImplementedError

                    SIGNALS[yvs[0]].update({"type": "const", "const": xt.shape[axis]})
                elif (kind == "aten::unbind"):
                    xt = SIGNALS[xvs[0]]["const"]
                    dim = SIGNALS[xvs[1]]["const"]
                    ndims = len(xt.shape)

                    if (ndims == 1):
                        yt = tf.squeeze(xt, axis=dim)
                    else:
                        raise NotImplementedError(f"Unable to handle {ndims}D tensors")

                    SIGNALS[yvs[0]].update({"type": "const", "const": yt})
                elif (kind in ["prim::NumToTensor", "aten::ScalarImplicit"]):
                    num = SIGNALS[xvs[0]]["const"]
                    SIGNALS[yvs[0]].update({"type": "const", "const": num})
                elif (kind == "aten::Int"):
                    num = SIGNALS[xvs[0]]["const"]
                    SIGNALS[yvs[0]].update({"type": "const", "const": int(num)})
                elif (kind == "aten::floor"):
                    xt = SIGNALS[xn.input()]["const"]
                    SIGNALS[yvs[0]].update({"type": "const", "const": np.floor(xt)})
                elif (kind == "aten::floor_divide"):
                    if (SIGNALS[xvs[0]]["type"] == "const"):
                        num = SIGNALS[xvs[0]]["const"]
                        divisor = SIGNALS[xvs[1]]["const"]
                        SIGNALS[yvs[0]].update({"type": "const", "const": np.floor(num / divisor)})
                    else:
                        raise NotImplementedError
                elif (kind in ["aten::to", "aten::detach"]):
                    xt = SIGNALS[xvs[0]]["const"]
                    SIGNALS[yvs[0]].update({"type": "const", "const": xt})
                elif (kind in ["prim::Constant", "prim::GetAttr"]):  # layers removed
                    pass
                else:
                    print(f"[E] Unknown kind='{kind}' provided.\n  {xn.schema()}\n", flush=True)
                    raise NotImplementedError

                ##  Update the termination condition.
                if (xn in output_nodes):  output_nodes.remove(xn)
            except StopIteration:
                break
        print()

        ##  Create a Keras model
        OUTPUTS = [SIGNALS[_]["tensor"] for _ in self.graph_outputs]
        if (type(OUTPUTS[0]) == list):  OUTPUTS = OUTPUTS[0]
        OUTPUTS.sort(key=lambda _: _.shape[1] * _.shape[2])
        keras_model = keras.Model(INPUTS, OUTPUTS, name=self.model_name)

        temp_dir = tempfile.mkdtemp()
        temp_file = f"{temp_dir}/{self.model_name}.keras"
        keras_model.save(temp_file)  # clean up the model graph.
        keras_model = keras.models.load_model(temp_file)
        os.system(f"rm -rf {temp_dir}")
        if (log_file):
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        return keras_model

    def _convert_torch2torch(self, log_file):
        params = self.params
        SIGNALS = self.SIGNALS
        Layer_info = []  # format :: {input.debugName , output.debugName,  ScopeName, Layer, state: 'pass', 'module', 'operator'}

        if (log_file):
            print(f"[I] Torch-to-Torch conversion log is written to '{log_file}'.")
            sys.stdout = open(log_file, "w")

        if (self.input_names):
            _xniter = self.graph.nodes()
            while (True):
                try:
                    xn = next(_xniter)
                    if (xn == self.graph_inputs[0].node()): break
                except StopIteration:
                    break

            yv = xn.output()
            found = re.findall(fr"^%{yv.debugName()} : .*\((.*), strides", str(xn))[0]
            input_shape = [int(_) for _ in found.split(",")]
        else:
            yv = self.graph_inputs[0]
            input_shape = self.input_shape
            _xniter = self.graph.nodes()

        print(f"[I] Convert torch Graph operators to Torch layers. \n")
        print("  {:24}{:20}{}".format("Layer", "Output", "Shape"))
        print("  " + "-" * 68)
        print("  {:24}{:20}{}".format("Input", yv.debugName(), input_shape))

        ## Process hidden layers.

        output_nodes = [_.node() for _ in self.graph_outputs]

        while (len(output_nodes) > 0):
            try:

                xn = next(_xniter)
                kind = xn.kind()
                xvs = [xn.inputsAt(_) for _ in range(xn.inputsSize())]
                yvs = [xn.outputsAt(_) for _ in range(xn.outputsSize())]


                if (kind == "aten::_convolution"):


                    stride         = SIGNALS[xvs[3]]["const"]  # stride
                    padding        = SIGNALS[xvs[4]]["const"]  # padding size
                    dilation       = SIGNALS[xvs[5]]["const"]  # dilation factor
                    groups         = SIGNALS[xvs[8]]["const"]  # groups

                    scope          = re.sub(r"^.*/__module\.", "", xn.scopeName())  #
                    var_name       = re.sub(r"\.","_", scope)                       # module scope name
                    k              = params.get(f"{scope}.weight")                  # weight
                    YC, XC, KH, KW = k.size()                                       # kernel size
                    depthwise      = (groups > 1 and groups == YC)
                    b              = params.get(f"{scope}.bias")                    # bias
                    bias           = (b != None)                                    # use bias or not


                    if(depthwise):
                        kwargs = {
                            "in_channels": YC,
                            "out_channels": YC,
                            "kernel_size": (KH, KW),
                            "stride": stride,
                            "padding": padding,
                            "dilation": dilation,
                            "groups": groups,
                            "bias"  : bias
                        }
                    else:
                        kwargs = {
                            "in_channels": XC,
                            "out_channels": YC,
                            "kernel_size": (KH, KW),
                            "stride": stride,
                            "padding": padding,
                            "dilation": dilation,
                            "groups": groups,
                            "bias"  : bias
                        }


                    inp_tensor          = re.sub(r"\.", "_",xvs[0].debugName())

                    if inp_tensor.isdigit() ==False:
                        inp_tensor      = '_'+ inp_tensor


                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())

                    if out_tensor.isdigit():                      # input tensor name is digit then add _ front of name
                        out_tensor      ='_'+out_tensor


                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Conv2d(**kwargs)}", 'state': 'module'})

                elif (kind == "aten::max_pool2d"):

                    (KH, KW)            = SIGNALS[xvs[1]]["const"]
                    padding             = SIGNALS[xvs[2]]["const"]
                    stride              = SIGNALS[xvs[3]]["const"]
                    dilation            = SIGNALS[xvs[4]]["const"]
                    ceil_mode           = SIGNALS[xvs[5]]["const"]


                    kwargs = {
                        'kernel_size': (KH, KW),
                        'padding': tuple(padding),
                        'stride': tuple(stride),
                        'dilation': tuple(dilation),
                        'ceil_mode': ceil_mode,
                    }
                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)


                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())
                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor



                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.MaxPool2d(**kwargs)}", 'state': 'module'})

                elif (kind == "aten::adaptive_avg_pool2d"):

                    output_size         = SIGNALS[xvs[1]]["const"]
                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())
                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor


                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.AdaptiveAvgPool2d(output_size)}", 'state': 'module'})

                elif (kind == "aten::batch_norm"):


                    momentum            = SIGNALS[xvs[6]]["const"]
                    epsilon             = SIGNALS[xvs[7]]["const"]
                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)

                    k                   = params[f"{scope}.weight"]
                    b                   = params[f"{scope}.bias"]
                    r_m                 = params[f"{scope}.running_mean"]
                    r_v                 = params[f"{scope}.running_var"]

                    affine              = (k != None  ) and (b != None)   # weight and bias are available

                    track_running_stats = (r_m !=None) and (r_v != None)  # running mean and variance are available


                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.BatchNorm2d(num_features=k.size()[0], eps=epsilon, momentum=momentum, affine=affine, track_running_stats=track_running_stats)}",
                         'state': 'module'})


                elif (kind == "aten::view"):

                    _size               = SIGNALS[xvs[1]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())
                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor


                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.view({_size})", 'state': 'operator'})

                elif (kind == "aten::reshape"):

                    _size               = SIGNALS[xvs[1]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor


                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.reshape({_size})", 'state': 'operator'})

                elif (kind == "aten::flatten"):

                    start_dim           = SIGNALS[xvs[1]]["const"]
                    end_dim             = SIGNALS[xvs[2]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.flatten(start_dim={start_dim},end_dim= {end_dim})",
                         'state': 'operator'})


                elif (kind == "aten::contiguous"):

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor


                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.contiguous()",
                         'state': 'operator'})

                elif (kind == "aten::transpose"):

                    dim0                = SIGNALS[xvs[1]]["const"]
                    dim1                = SIGNALS[xvs[2]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.transpose({dim0},{dim1})",
                         'state': 'operator'})

                elif (kind == "aten::permute"):

                    _dims               = SIGNALS[xvs[1]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.permute({_dims})",
                         'state': 'operator'})

                elif (kind == "aten::split_with_sizes"):

                    split_sizes         = SIGNALS[xvs[1]]["const"]
                    _dim                = SIGNALS[xvs[2]]["const"]


                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.split({split_sizes}, dim ={_dim})",
                         'state': 'operator'})


                elif (kind == "aten::chunk"):

                    chunks              = SIGNALS[xvs[1]]["const"]
                    _dim                = SIGNALS[xvs[2]]["const"]

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"torch.chunk({inp_tensor},{chunks},dim={_dim})",
                         'state': 'operator'})



                elif (kind == "aten::upsample_nearest2d"):


                    scale_factors       = SIGNALS[xvs[2]]["const"]
                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Upsample(scale_factor=scale_factors[0], mode='nearest')}", 'state': 'module'})

                elif (kind == "aten::upsample_bilinear2d"):

                    align_corners       = SIGNALS[xvs[2]]["const"]
                    scale_factors       = SIGNALS[xvs[3]]["const"]

                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)
                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Upsample(scale_factor=scale_factors[0], mode='bilinear', align_corners=align_corners)}",
                         'state': 'module'})

                elif (kind == "aten::cat"):

                    _dim                = SIGNALS[xvs[1]]["const"]

                    inp                 = re.findall('\%([^\:\,\)]+)', str(xvs[0].debugName))
                    inp_tensor          = ''

                    for _ in range(1, len(inp)):
                        inp_tensor     +=  '_'+re.sub(r"\.", "_", inp[_]) +','  if  re.sub(r"\.", "_", inp[_]).isdigit() else re.sub(r"\.", "_", inp[_]) +','

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"torch.cat(({inp_tensor}),{_dim})",
                         'state': 'operator'})




                elif (kind in ["aten::add", "aten::add_"]):

                    alpha               = SIGNALS[xvs[2]]["const"]


                    inp_tensor1         = re.sub(r"\.", "_", xvs[0].debugName())

                    inp_tensor2         = re.sub(r"\.", "_", xvs[1].debugName())



                    if inp_tensor1.isdigit():
                        inp_tensor1     = '_' + inp_tensor1
                    if inp_tensor2.isdigit():
                        inp_tensor2     = '_' + inp_tensor2

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append({'input': (inp_tensor1, inp_tensor2), 'output': out_tensor,
                                       'scope': xn.scopeName(),
                                       'Layer': f"torch.add({inp_tensor1},{inp_tensor2},alpha={alpha})",
                                       'state': 'operator'})



                elif (kind == "aten::mul"):

                    inp_tensor1         = re.sub(r"\.", "_", xvs[0].debugName())

                    inp_tensor2         = re.sub(r"\.", "_", xvs[1].debugName())

                    if inp_tensor1.isdigit():
                        inp_tensor1     = '_' + inp_tensor1
                    if inp_tensor2.isdigit():
                        inp_tensor2     = '_' + inp_tensor2

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': (inp_tensor1, inp_tensor2), 'output': out_tensor,
                         'scope': xn.scopeName(),
                         'Layer': f"torch.mul({inp_tensor1},{inp_tensor2})",
                         'state': 'operator'})

                elif (kind == "aten::relu_"):

                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.ReLU()}", 'state': 'module'})

                elif (kind == "aten::sigmoid"):

                    scope               = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name            = re.sub(r"\.", "_", scope)

                    inp_tensor          = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor      = '_' + inp_tensor

                    out_tensor          = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor      = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Sigmoid()}", 'state': 'module'})

                elif (kind == "aten::hardsigmoid"):

                    scope              = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name           = re.sub(r"\.", "_", scope)

                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Hardsigmoid()}", 'state': 'module'})

                elif (kind == "aten::silu_"):


                    scope              = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name           = re.sub(r"\.", "_", scope)
                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.SiLU()}", 'state': 'module'})
                elif (kind == "aten::hardswish"):

                    scope              = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name           = re.sub(r"\.", "_", scope)

                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.Hardswish()}", 'state': 'module'})

                elif (kind == "aten::leaky_relu_"):

                    alpha              = SIGNALS[xvs[1]]["const"]

                    scope              = re.sub(r"^.*/__module\.", "", xn.scopeName())
                    var_name           = re.sub(r"\.", "_", scope)

                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': var_name,
                         'Layer': f"nn.{nn.LeakyReLU(alpha)}", 'state': 'module'})


                elif (kind in ["prim::ListConstruct", "prim::TupleConstruct"]):

                    pass

                elif (kind in ["prim::ListUnpack", "prim::TupleUnpack"]):

                    inp                = re.findall('\%([^\:\,\)]+)', str(xvs[0].debugName))

                    out                = re.findall('\%([^\s\:\)]+)',str(yvs[0].debugName))

                    inp_tensor         = ''
                    out_tensor         = ''

                    for _ in range(1, len(inp)):
                        inp_tensor    += '_' + re.sub(r"\.", "_", inp[_]) + ',' if re.sub(r"\.", "_",inp[_]).isdigit() else re.sub(r"\.", "_", inp[_]) + ','
                    for _ in range(0,len(out)-1):
                        out_tensor    += '_' + re.sub(r"\.", "_", out[_]) + ',' if re.sub(r"\.", "_",out[_]).isdigit() else re.sub(r"\.", "_", out[_]) + ','

                    Layer              = out_tensor + '=' + inp_tensor

                    Layer_info.append(
                        {'input': xvs[0].debugName(), 'output': (yvs[i].debugName() for i in range(len(yvs))),
                         'scope': xn.scopeName(), 'Layer': f"{Layer}",
                         'state': 'pass'})

                elif (kind in ["aten::size"]):

                    _axis              = SIGNALS[xvs[1]]["const"]


                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())
                    if inp_tensor.isdigit():
                        inp_tensor     = '_'+ inp_tensor


                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_'+out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}.size(dim={_axis})",
                         'state': 'operator'})


                elif (kind == "aten::unbind"):

                    dim                = SIGNALS[xvs[1]]["const"]

                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"torch.unbind({inp_tensor},dim={dim})",
                         'state': 'operator'})


                elif (kind in ["prim::NumToTensor", "aten::ScalarImplicit"]):

                    pass

                elif (kind == "aten::Int"):

                    pass

                elif (kind == "aten::floor"):

                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"torch.floor({inp_tensor})",
                         'state': 'operator'})



                elif (kind == "aten::floor_divide"):


                    divisor            = SIGNALS[xvs[1]]["const"]


                    inp_tensor         = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor     = '_' + inp_tensor

                    out_tensor         = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor     = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"torch.floor_divide({inp_tensor},{divisor})",
                         'state': 'operator'})


                elif (kind in ["aten::to", "aten::detach"]):


                    inp_tensor        = re.sub(r"\.", "_", xvs[0].debugName())

                    if inp_tensor.isdigit():
                        inp_tensor    = '_' + inp_tensor

                    out_tensor        = re.sub(r"\.", "_", yvs[0].debugName())
                    if out_tensor.isdigit():
                        out_tensor    = '_' + out_tensor

                    Layer_info.append(
                        {'input': inp_tensor, 'output': out_tensor, 'scope': xn.scopeName(),
                         'Layer': f"{inp_tensor}={out_tensor}",
                         'state': 'pass'
                         }
                    )

                elif (kind in ["prim::Constant", "prim::GetAttr"]):  # layers removed

                    pass

                else:

                    print(f"[E] Unknown kind='{kind}' provided.\n  {xn.schema()}", flush=True)

                    raise NotImplementedError

                ##  Update the termination condition.

                if (xn in output_nodes):  output_nodes.remove(xn)

            except StopIteration:
                break

        print()

        ##########    model.py generator
        """
        Layer Info : {input.debugName , output.debugName,  ScopeName, Layer, state: ('pass', 'module', 'operator')}
        """
        model_file = "./torch_model.py"
        f = open(model_file, 'w')
        f.write(f"import torch\n")
        f.write(f"import torch.nn as nn\n\n\n\n")
        f.write(f"class Model(nn.Module):\n")
        f.write(f"\t" + "def __init__(self):\n")
        f.write(f"\t" * 2 + "super().__init__()\n\n")
        for _layer_info in Layer_info:
            if (_layer_info['state'] == 'module'):
                f.write(f"\t" * 2 + "self." + f"{_layer_info['scope']} = {_layer_info['Layer']}\n")

        f.write(f"\n\t" + f"def forward(self, {Layer_info[0]['input']}):\n\n")

        for _layer_info in Layer_info:
            if _layer_info['state'] == 'module':
                f.write(
                    f"\t" * 2 + f"{_layer_info['output']}=" + f"self.{_layer_info['scope']}({_layer_info['input']})\n")
            elif _layer_info['state'] == 'operator':
                f.write(f"\t" * 2 + f"{_layer_info['output']}=" + f"{_layer_info['Layer']}\n")
            else:
                f.write(f"\t" * 2 + f"{_layer_info['Layer']}\n")

        f.write(f"\n" + "\t" * 2 + f"return {Layer_info[-1]['output']}\n")


        param_file = f'./parameter/new_params_{self.model_name}.pt'



        ### parameter key renaming
        new_params = OrderedDict([('_'.join(k.split('.')[:-1])+'.'+k.split('.')[-1],v) for k,v in params.items()])

        torch.save(new_params, param_file)

        return model_file, param_file

    @staticmethod
    def _optimize_yolox(model, mosaic=True):
        """    Optimize the YoloX-Nano network model

        (1) Fuse the input mosaic to the first Conv2D layer, which becomes 6x6 strided Conv2d.
        (2) Reorder the color channel from BGR to RGB.
        (3) Normalize the input range to [0,1) (preproc="raw").
        """
        SIGNALS = {}  # signal dictionary accessed by "layer" key
        OUTPUTS = []
        print(f"[I] Optimize the model '{model.name}'.")

        for L0 in model.layers:
            assert len(L0.inbound_nodes) == 1, "Not Implemented"
            split = None  # output tensor split information
            xn = L0.inbound_nodes[0]
            LN1s = xn.inbound_layers  # previous layers

            ##  Generate the layer outputs.
            if (hasattr(L0, "is_placeholder") and L0.is_placeholder):
                if (mosaic):
                    INPUT_SHAPE = (model.input_shape[1] * 2, model.input_shape[2] * 2, model.input_shape[3] // 4)
                else:
                    INPUT_SHAPE = model.input_shape[1:]

                INPUTS = keras.layers.Input(INPUT_SHAPE, name=model.input_names[0])
                yt = INPUTS
            elif (isinstance(L0, keras.layers.Conv2D) and model.input_shape == L0.input_shape):
                ##  Fuse the image processing into the first Conv2D.
                ##  Reorder the color channels from BGR to RGB.
                KH, KW, XC, YC = L0.weights[0].shape

                if (mosaic):
                    _K12 = np.transpose(L0.weights[0], (3, 0, 1, 2))  # HWCN-to-NHWC
                    _K03 = np.ndarray((YC, KH * 2, KW * 2, XC // 4), dtype=np.float32)
                    for yc in range(YC):
                        for kh in range(KH):
                            for kw in range(KW):
                                _K03[yc][kh * 2][kw * 2] = _K12[yc][kh][kw][[2, 1, 0]] * 256
                                _K03[yc][kh * 2][kw * 2 + 1] = _K12[yc][kh][kw][[8, 7, 6]] * 256
                                _K03[yc][kh * 2 + 1][kw * 2] = _K12[yc][kh][kw][[5, 4, 3]] * 256
                                _K03[yc][kh * 2 + 1][kw * 2 + 1] = _K12[yc][kh][kw][[11, 10, 9]] * 256
                    K = _K03.transpose((1, 2, 3, 0))  # NHWC-to-HWCN
                    kernel_size = (KH * 2, KW * 2)
                    strides = [_ * 2 for _ in L0.strides]
                else:
                    K = L0.weights[0] * 256
                    kernel_size = (KH, KW)
                    strides = L0.strides

                ##  Generate the output.
                xv = SIGNALS[LN1s]
                yt = keras.layers.Conv2D(
                    YC, kernel_size,
                    strides=strides,
                    padding=L0.padding,
                    use_bias=L0.use_bias,
                    activation=L0.activation,
                    weights=[K, L0.weights[1]] if (L0.use_bias) else [K]
                )(xv["tensor"])

            elif (hasattr(L0, "function")):  # [TensorFlow functions]
                if (type(LN1s) == list):  raise NotImplementedError

                xv = SIGNALS[LN1s]
                yt = L0(xv["tensor"], **xn.call_kwargs)

                if (L0.function.__name__ == "split"):
                    split = {}  # split tensors must be stacked in reverse order.
                    nsplits = len(L0.outbound_nodes)
                    for i, onode in enumerate(L0.outbound_nodes):
                        split[onode] = nsplits - 1 - i

            else:  # [Keras layer classes]
                if (type(LN1s) == list):
                    xts = []
                    for LN1 in LN1s:
                        xv = SIGNALS[LN1]
                        if (xv["split"] != None):
                            xts.append(xv["tensor"][xv["split"][xn]])
                        else:
                            xts.append(xv["tensor"])
                    yt = L0(xts)
                else:
                    xv = SIGNALS[LN1s]
                    if (xv["split"] != None):
                        yt = L0(xv["tensor"][xv["split"][xn]])
                    else:
                        yt = L0(xv["tensor"])
            ##  End of if-elif-else blocks

            ##  Update the output dictionary to build the model.
            SIGNALS[L0] = {
                "name": L0.name,  # layer name
                "split": split,  # output tensor split information
                "tensor": yt  # output tensor
            }
            if (L0.name in model.output_names): OUTPUTS.append(yt)
        ##  End of for (model.layers)

        return keras.Model(INPUTS, OUTPUTS, name=model.name)

    @staticmethod
    def _optimize_nanodet(model):
        """    Optimize the NanoDet network model.

        (1) Reorder the color channel from BGR to RGB.
        (2) Normalize the input range to [0,1) (preproc="raw").
        """
        SIGNALS = {}  # signal dictionary accessed by "layer" key
        OUTPUTS = []
        print(f"[I] Optimize the model '{model.name}'.")

        for L0 in model.layers:
            assert len(L0.inbound_nodes) == 1, "Not Implemented"
            split = None  # output tensor split information
            xn = L0.inbound_nodes[0]
            LN1s = xn.inbound_layers  # previous layers

            if (hasattr(L0, "is_placeholder") and L0.is_placeholder):
                INPUTS = keras.layers.Input(model.input_shape[1:], name=model.input_names[0])
                yt = INPUTS
            elif (isinstance(L0, keras.layers.Conv2D) and L0.input_shape[3] == 3):
                ##  Reorder the color channels from BGR to RGB.
                _K0 = np.transpose(L0.weights[0], (3, 0, 1, 2))  # HWCN-to-NHWC
                _K0[:, :, :] = _K0[:, :, :, [2, 1, 0]]

                ##  Fuse the image processing into the first Conv2D and BatchNormalizaiton layers.
                mean = np.array([0.4831, 0.4542, 0.4044], dtype="float32")
                sigma = np.array([0.2281, 0.2231, 0.2241], dtype="float32")
                _B2_shift = np.sum((mean / sigma) * np.sum(_K0, axis=(1, 2)), 1)
                _K0 /= sigma

                K = _K0.transpose((1, 2, 3, 0))  # NHWC-to-HWCN
                L0.set_weights([K, L0.weights[1]] if (L0.use_bias) else [K])
                L1 = L0.outbound_nodes[0].outbound_layer
                B = L1.get_weights()
                B[2] += _B2_shift
                L1.set_weights(B)

                ##  Generate the output.
                xv = SIGNALS[LN1s]
                yt = L0(xv["tensor"])

            elif (hasattr(L0, "function")):  # [TensorFlow functions]
                if (type(LN1s) == list):  raise NotImplementedError
                xv = SIGNALS[LN1s]
                yt = L0(xv["tensor"], **xn.call_kwargs)

                if (L0.function.__name__ == "split"):
                    split = {}  # split tensors must be stacked in reverse order.
                    nsplits = len(L0.outbound_nodes)
                    for i, onode in enumerate(L0.outbound_nodes):
                        split[onode] = nsplits - 1 - i

            else:  # [Keras layer classes]
                if (type(LN1s) == list):
                    xts = []
                    for LN1 in LN1s:
                        xv = SIGNALS[LN1]
                        if (xv["split"] != None):
                            xts.append(xv["tensor"][xv["split"][xn]])
                        else:
                            xts.append(xv["tensor"])
                    yt = L0(xts)
                else:
                    xv = SIGNALS[LN1s]
                    if (xv["split"] != None):
                        yt = L0(xv["tensor"][xv["split"][xn]])
                    else:
                        yt = L0(xv["tensor"])
            ##  End of if-elif-else blocks

            ##  Update the output dictionary to build the model.
            SIGNALS[L0] = {
                "name": L0.name,  # layer name
                "split": split,  # output tensor split information
                "tensor": yt  # output tensor
            }
            if (L0.name in model.output_names): OUTPUTS.append(yt)
        ##  End of for (model.layers)

        return keras.Model(INPUTS, OUTPUTS, name=model.name)

    def __repr__(self):
        return f"""{__class__.__name__}
  model         : {type(self.model)}
  model_name    : {self.model_name}
  scriptmodule  : {type(self.scriptmodule)}
  graph         : {type(self.graph)}
  params        : {type(self.params)}, #={len(self.params)}
  input_shape   : {str(self.input_shape).replace(" ", "")}
  input_names   : {self.input_names[0] if (self.input_names) else self.graph_inputs[0].debugName()}
  output_names  : {self.output_names if (self.output_names) else [_.debugName() for _ in self.graph_outputs]}
  probe_names   : {self.probe_names}"""

    def __str__(self):
        return self.__repr__()

##  End of class TorchManager
