import ctypes
import functools
import logging
import sys

import torch
import torch.nn.functional as F

import comfy.ops
from comfy.ldm.krea2.model import SwiGLU

from .hip_attention import Kernels, launch

BLOCK_ROWS = 512  # WAVES * 64 in int8_gemm.hip
ROT_THREADS = 128  # ROT_THREADS in int8_gemm.hip
MIN_ROWS = 512  # Below this the weight packing costs more than the kernel saves.
MAX_ROT_K = 16384  # Keeps the rotated row within the LDS of a block.
ROT_GROUP = 256

KERNELS = ("quant_pack_a", "rot_quant_pack_a", "situ_rot_quant_pack_a", "swiglu_rot_quant_pack_a", "rot_quant_pack_a_i4", "pack_b", "gemm", "gemm_i4")

kernels = None
originals = {}
# ConvRot runs before the linear. rotate_activation leaves the rotation to the kernel by handing the tensor over here.
fusable = False
pending = None
# Set by linear_input_act when the next linear gets the raw [hidden | gate] input of a situ activation.
situ_next = False
# Set when the next linear gets a SwiGLU gate as its input, with this as the up projection.
swiglu_up = None
# The Int8 layer only reaches wrap through the Triton path of int8_quant.
swiglu_fusable = False


def get_kernels():
    global kernels
    if kernels is None:
        kernels = Kernels("int8_gemm", KERNELS)
    return kernels


def run(quant, gemm, x, weight, weight_scale, bias, K, up=None):
    """K is the number of values per row. weight is (N, row_bytes) int8 and weight_scale (N,) fp32. Returns (M, N) fp16."""
    M = x.shape[0]
    N, row_bytes = weight.shape
    blocks = (M + BLOCK_ROWS - 1) // BLOCK_ROWS
    packed_a = torch.empty(blocks * BLOCK_ROWS * row_bytes // 4, dtype=torch.int32, device=x.device)
    # gemm prefetches one step past the end of the weights.
    packed_b = torch.empty(N * row_bytes // 4 + 1024, dtype=torch.int32, device=x.device)
    a_scale = torch.empty(M, dtype=torch.float32, device=x.device)
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    rotates = quant is not get_kernels().quant_pack_a
    inputs = [c_ptr(x.data_ptr())] if up is None else [c_ptr(x.data_ptr()), c_ptr(up.data_ptr())]
    launch(quant, M, 1, ROT_THREADS if rotates else 256, [*inputs, c_ptr(packed_a.data_ptr()), c_ptr(a_scale.data_ptr()), c_int(K)], K * 2 if rotates else 0)
    launch(get_kernels().pack_b, row_bytes // 64, N // 64, 256, [c_ptr(weight.data_ptr()), c_ptr(packed_b.data_ptr()), c_int(row_bytes)])
    launch(gemm, blocks, N // 64, 256, [
        c_ptr(packed_a.data_ptr()), c_ptr(packed_b.data_ptr()), c_ptr(a_scale.data_ptr()), c_ptr(weight_scale.data_ptr()),
        c_ptr(bias.data_ptr() if bias is not None else None), c_ptr(out.data_ptr()), c_int(M), c_int(N), c_int(row_bytes // 64)])
    return out


def linear(x, weight, weight_scale, bias, rotate=False, situ=False, up=None):
    """x is (M, K) fp16, or (M, 2K) with situ, weight (N, K) int8 and weight_scale (N,) fp32. With up, x is the gate of a SwiGLU. Returns (M, N) fp16."""
    k = get_kernels()
    quant = k.swiglu_rot_quant_pack_a if up is not None else k.situ_rot_quant_pack_a if situ else k.rot_quant_pack_a if rotate else k.quant_pack_a
    return run(quant, k.gemm, x, weight, weight_scale, bias, weight.shape[1], up)


def wrap_w4a4(original):
    def convrot_w4a4_forward(x, qweight, wscales, bias, convrot_groupsize, compute_dtype, skip_rotation=False, qweight_unpacked=None, x_rot=None, orig_shape=None):
        N, row_bytes = qweight.shape
        K = row_bytes * 2
        if (x_rot is not None or skip_rotation or convrot_groupsize != ROT_GROUP or compute_dtype != torch.float16 or x.shape[-1] != K or K > MAX_ROT_K
                or x.numel() // K < MIN_ROWS or N % 64 != 0 or K % 128 != 0 or (bias is not None and bias.dtype != torch.float16)):
            return original(x, qweight, wscales, bias, convrot_groupsize, compute_dtype, skip_rotation, qweight_unpacked, x_rot, orig_shape)

        k = get_kernels()
        x2d = x.reshape(-1, K).to(torch.float16).contiguous()
        out = run(k.rot_quant_pack_a_i4, k.gemm_i4, x2d, qweight.contiguous(), wscales.reshape(N).float().contiguous(), bias, K)
        return out.reshape(x.shape[:-1] + (N,))
    return convrot_w4a4_forward


def wrap(original):
    def int8_linear(x, weight, weight_scale, bias=None, compute_dtype=torch.float16, x_q=None):
        global pending, situ_next, swiglu_up
        rotate, situ, up = x is pending, situ_next, swiglu_up
        pending, situ_next, swiglu_up = None, False, None
        N, K = weight.shape
        if (x_q is not None or x.dtype != torch.float16 or compute_dtype != torch.float16 or (bias is not None and bias.dtype != torch.float16)
                or x.numel() // x.shape[-1] < MIN_ROWS or N % 64 != 0 or K % 64 != 0):
            if situ:
                x = originals["eager_input_act"](x, "situ")
            if up is not None:
                x = F.silu(x).mul_(up)
            if rotate:
                x = originals["rotate_activation"](x, originals["build_hadamard"](ROT_GROUP, device=x.device, dtype=x.dtype), ROT_GROUP)
            return original(x, weight, weight_scale, bias, compute_dtype, x_q)

        if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() == N:
            scale = weight_scale.reshape(N).float().contiguous()
        else:
            scale = torch.full((N,), float(weight_scale), dtype=torch.float32, device=x.device)
        out = linear(x.reshape(-1, x.shape[-1]).contiguous(), weight.contiguous(), scale, bias, rotate, situ, None if up is None else up.reshape(-1, K).contiguous())
        return out.reshape(x.shape[:-1] + (N,))
    return int8_linear


def rotate_activation(x, H, group_size):
    global pending
    if situ_next or swiglu_up is not None:
        pending = x
        return x
    if fusable and group_size == ROT_GROUP and x.dtype == torch.float16 and x.dim() == 2 and x.shape[1] % ROT_GROUP == 0 and x.shape[1] <= MAX_ROT_K:
        pending = x
        return x
    return originals["rotate_activation"](x, H, group_size)


def fuse_act(layer, x, K):
    """Only for layers where the kernel can do the activation, ConvRot and the quantization in one pass. K is the width of the activation."""
    return (getattr(layer, "_is_quantized", False) and getattr(layer, "_use_convrot", False) and not getattr(layer, "_is_w4a8", False)
            and getattr(layer, "_convrot_groupsize", 0) == ROT_GROUP and not layer.lora_patches and x.dtype == torch.float16
            and K == layer.in_features and layer.in_features % ROT_GROUP == 0 and layer.in_features <= MAX_ROT_K)


def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0, residual=None, residual_scale=None):
    global situ_next
    if input_act == "situ" and residual is None and fuse_act(linear, x, x.shape[-1] // 2):
        situ_next = True
        return linear(x)
    return originals["linear_input_act"](linear, x, input_act, act_weight, act_eps, residual, residual_scale)


def swiglu_forward(module, x):
    global swiglu_up
    gate, up = module.gate(x), module.up(x)
    # Int4 layers have the same flags but never reach wrap.
    if swiglu_fusable and isinstance(module.down, originals["layer"]) and fuse_act(module.down, gate, gate.shape[-1]):
        swiglu_up = up
        return module.down(gate)
    return module.down(F.silu(gate).mul_(up))


def patch_swiglu(model):
    for name, module in model.get_model_object("diffusion_model").named_modules():
        if isinstance(module, SwiGLU):
            model.add_object_patch(f"diffusion_model.{name}.forward", functools.partial(swiglu_forward, module))


def wrap_forward(forward):
    def int8_forward(self, x):
        # Dynamic LoRA reads the rotated activation after the linear, so those layers keep the separate rotation.
        global fusable
        fusable = not self.lora_patches
        try:
            return forward(self, x)
        finally:
            fusable = False
    return int8_forward


def enable(on):
    global swiglu_fusable
    pack = next((m for name, m in sys.modules.items() if name.endswith(".int8_quant") and hasattr(m, "triton_int8_linear_per_row")), None)
    int4 = next((m for name, m in sys.modules.items() if name.endswith(".int4_quant") and hasattr(m, "convrot_w4a4_forward")), None)
    if int4 is not None:
        originals.setdefault("convrot_w4a4_forward", int4.convrot_w4a4_forward)
        int4.convrot_w4a4_forward = wrap_w4a4(originals["convrot_w4a4_forward"]) if on else originals["convrot_w4a4_forward"]
    if pack is None:
        if on:
            logging.warning("RDNA2 Attention: int8_gemm skipped, ComfyUI-INT8-Fast-ROCM is not loaded.")
        return

    convrot = next(m for name, m in sys.modules.items() if name.endswith(".convrot") and hasattr(m, "rotate_activation"))
    layer = pack.Int8TensorwiseOps.Linear
    if "forward" not in originals:
        originals.update(layer=layer, triton_int8_linear=pack.triton_int8_linear, triton_int8_linear_per_row=pack.triton_int8_linear_per_row,
                         rotate_activation=convrot.rotate_activation, build_hadamard=convrot.build_hadamard, forward=layer.forward,
                         linear_input_act=comfy.ops.linear_input_act, eager_input_act=comfy.ops._eager_input_act)

    for name in ("triton_int8_linear", "triton_int8_linear_per_row"):
        setattr(pack, name, wrap(originals[name]) if on else originals[name])
    convrot.rotate_activation = rotate_activation if on else originals["rotate_activation"]
    layer.forward = wrap_forward(originals["forward"]) if on else originals["forward"]
    comfy.ops.linear_input_act = linear_input_act if on else originals["linear_input_act"]
    swiglu_fusable = on and pack._TRITON_AVAILABLE
