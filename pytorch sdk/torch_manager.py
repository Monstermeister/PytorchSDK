#!/usr/bin/env python3

import torch
import torch2torch


if (__name__=="__main__"):
  ##  Save a Torch model for easy loading.
  """
  import yolox.exp

  model         = yolox.exp.get_exp(None, "yolox_nano").get_model()
  checkpoint    = torch.load("parameter/yolox_nano.pth", "cpu")
  model.load_state_dict(checkpoint["model"])
  torch.save(model, "parameter/yolox_nano.pt")
  exit()
  """

  ##  Convert the model framework from Torch to Keras.
  #model       = torch.load("parameter/yolox_nano.pt")
  model       = torch.load("parameter/yolox_nano.pt")
  manager     = torch2torch.TorchManager(
    model, (3,416,416),
    model_name="yolox_nano",
    input_names=["input.1"],            # custom input tensor
    output_names=["1942"],              # custom output tensor
    #probe_names=["input.53"]            # probes
  )
  manager.dump_graph("graph.log", output_dir="./output")
  #keras_model = manager.generate_keras(
  #  summary_file="yolox_nano.summary",
  #  keras_file="yolox_nano.keras",
  #  tflite_file="yolox_nano.tflite",
  #  log_file="convert.log",
  #  output_dir="./output"
  #)
  Torch_model, torch_param = manager._convert_torch2torch(None,)

  import torch_model

  test_model = torch_model.Model()
  test_model.load_state_dict(torch.load(torch_param))
