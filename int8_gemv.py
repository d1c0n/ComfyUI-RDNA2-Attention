import ctypes
import functools

import torch

import comfy.ops
from comfy.ops import MixedPrecisionOp
from comfy.text_encoders.llama import BaseQwen3

from .hip_attention import Kernels, launch

BLOCK_ROWS = 16  # WAVES * ROWS in int8_gemv.hip
STEP = 512  # STEP in int8_gemv.hip
BF16_STEP = 256  # BF16_STEP in int8_gemv.hip

kernels = None


def get_kernels():
    global kernels
    if kernels is None:
        kernels = Kernels("int8_gemv", ("rotate", "gemv_i8", "gemv_bf16"))
    return kernels


def linear_forward(module, N, K, weight_dtype, group, x, *args, **kwargs):
    if x.numel() != K or x.dtype != torch.float32 or module.weight_function or module.bias_function or getattr(module, "weight_lowvram_function", None) is not None:
        return type(module).forward(module, x, *args, **kwargs)

    k = get_kernels()
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    x = x.contiguous()
    with comfy.ops.CastBiasWeightContext(module, input=None, dtype=weight_dtype, device=x.device, bias_dtype=x.dtype,
                                         offloadable=True, compute_dtype=x.dtype, want_requant=True) as (weight, bias):
        if group:
            rotated = torch.empty_like(x)
            launch(k.rotate, K // group, 1, 256, [c_ptr(x.data_ptr()), c_ptr(rotated.data_ptr()), c_int(group)], group * 4)
            x = rotated
        out = torch.empty((*x.shape[:-1], N), dtype=torch.float32, device=x.device)
        scale = weight._params.scale.to(device=x.device)
        launch(k.gemv_i8, N // BLOCK_ROWS, 1, 256, [c_ptr(x.data_ptr()), c_ptr(weight._qdata.data_ptr()), c_ptr(scale.data_ptr()),
                                                    c_ptr(bias.data_ptr() if bias is not None else None), c_ptr(out.data_ptr()), c_int(K)], K * 4)
    return out


def logits(transformer, x):
    embed = transformer.model.embed_tokens
    if x.shape[0] != 1 or x.dtype != torch.float32 or embed.weight_function or getattr(embed, "weight_lowvram_function", None) is not None:
        return type(transformer).logits(transformer, x)

    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    x = x[:, -1:].contiguous()
    N, K = embed.weight.shape
    with comfy.ops.CastBiasWeightContext(embed, input=None, dtype=torch.bfloat16, device=x.device, offloadable=True) as (weight, _):
        out = torch.empty((1, 1, N), dtype=torch.float32, device=x.device)
        launch(get_kernels().gemv_bf16, N // BLOCK_ROWS, 1, 256, [c_ptr(x.data_ptr()), c_ptr(weight.data_ptr()), c_ptr(out.data_ptr()), c_int(K)], K * 4)
    return out


def patch(clip):
    """Single token steps of text generation read int8 and bf16 weights directly instead of dequantizing them on every token."""
    for name, module in clip.patcher.model.named_modules():
        if isinstance(module, MixedPrecisionOp) and type(module).__name__ == "Linear" and module.quant_format == "int8_tensorwise":
            N, K = module.weight.shape
            params = module.weight._params
            if N % BLOCK_ROWS == 0 and K % STEP == 0 and params.scale.numel() == N and params.scale.dtype == torch.float32 and not params.transposed:
                group = params.convrot_groupsize if params.convrot else 0
                clip.patcher.add_object_patch(f"{name}.forward", functools.partial(linear_forward, module, N, K, module.weight.dtype, group))
        elif isinstance(module, BaseQwen3) and not module.model.config.lm_head:
            embed = module.model.embed_tokens
            if embed.weight.dtype == torch.bfloat16 and embed.weight.shape[0] % BLOCK_ROWS == 0 and embed.weight.shape[1] % BF16_STEP == 0:
                clip.patcher.add_object_patch(f"{name}.logits", functools.partial(logits, module))
