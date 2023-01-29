import time
from options.export_options import ExportOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
import torch
import torchsummary
from thop import profile, clever_format
import numpy as np


opt = ExportOptions().parse()

model = create_model(opt)
model.setup(opt)

net_to_export = model.netG

h,w = tuple([int(x) for x in opt.export_size.strip().split(',')])
dummy_data = torch.as_tensor(
        np.random.normal(0, 1, (1, opt.input_nc, h, w)),
        dtype=torch.float32)
macs, params = profile(net_to_export, inputs=(dummy_data, ))
macs, params = clever_format([macs, params], "%.3f")

print('netG macs: ' + macs + ', # of params: '+params)

torch.onnx.export(
        net_to_export,
        (dummy_data,),
        'prototype.onnx',
        export_params=True,
        verbose=False,
        opset_version=11
    )


pass