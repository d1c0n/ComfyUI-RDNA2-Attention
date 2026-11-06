import ctypes
import functools

import torch

import comfy.model_management
from comfy.ldm.krea2.model import RMSNorm

from .hip_attention import Kernels, launch

THREADS = 256  # THREADS in rms_norm.hip
HEAD_DIM = 128  # 16 threads with 8 values each in rms_norm_head
MAX_DIM = 8192  # 256 threads with 4 x 8 values each in rms_norm_row

kernels = None


def rms_norm(x, weight, eps):
    """x is (B, L, D) or (B, H, L, D) fp16 with a contiguous last dimension, weight (D,) fp32 and zero-centered. Returns contiguous fp16."""
    global kernels
    if kernels is None:
        kernels = Kernels("rms_norm", ("rms_norm_head", "rms_norm_row"))

    v = x if x.ndim == 4 else x.unsqueeze(1)
    B, H, L, D = v.shape
    sb, sh, sl = v.stride()[:3]
    rows = B * H * L
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    head = D <= HEAD_DIM
    per_block = THREADS // 16 if head else 1
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    launch(kernels.rms_norm_head if head else kernels.rms_norm_row, (rows + per_block - 1) // per_block, 1, THREADS, [
        c_ptr(x.data_ptr()), c_ptr(weight.data_ptr()), c_ptr(out.data_ptr()), c_int(rows), c_int(D), c_int(L), c_int(H),
        c_int(sb), c_int(sh), c_int(sl), ctypes.c_float(eps)])
    return out


def forward(module, x):
    if x.dtype != torch.float16 or x.ndim not in (3, 4) or x.stride(-1) != 1 or x.shape[-1] % 8 != 0 or x.shape[-1] > MAX_DIM:
        return RMSNorm.forward(module, x)
    return rms_norm(x, comfy.model_management.cast_to(module.scale, dtype=torch.float32, device=x.device), module.eps)


def patch(model):
    for name, module in model.get_model_object("diffusion_model").named_modules():
        if isinstance(module, RMSNorm):
            model.add_object_patch(f"diffusion_model.{name}.forward", functools.partial(forward, module))
