# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()

def inference_model(model, imgs, dtype=torch.bfloat16):
    # forward the model
    with torch.no_grad():
        (results,) = model(imgs.to(dtype).cuda())
        imgs.cpu()

    return results

def load_model(checkpoint, use_torchscript=False):
    if use_torchscript:
        return torch.jit.load(checkpoint)
    else:
        return torch.export.load(checkpoint).module()

def sapiens(checkpoint, fp16=False):
    torch._inductor.config.force_fuse_int_mm_with_mul = True
    torch._inductor.config.use_mixed_mm = True
    USE_TORCHSCRIPT = '_torchscript' in checkpoint
    # build the model from a checkpoint file
    model = load_model(checkpoint, USE_TORCHSCRIPT)

    ## no precision conversion needed for torchscript. run at fp32
    if not USE_TORCHSCRIPT:
        dtype = torch.half if fp16 else torch.bfloat16
        model.to(dtype)
        model = torch.compile(model, mode="max-autotune", fullgraph=True)
    else:
        dtype = torch.float32  # TorchScript models use float32
        # model = model.to(device)

    return model
