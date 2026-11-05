import ctypes
import logging
import sys

import torch

import comfy.ops

from .hip_attention import Kernels, launch

BLOCK_ROWS = 512  # WAVES * 64 in int8_gemm.hip
MIN_ROWS = 512  # Below this the weight packing costs more than the kernel saves.
MAX_ROT_K = 16384  # Size of the row buffer in rot_quant_pack_a.
ROT_GROUP = 256

KERNELS = ("quant_pack_a", "rot_quant_pack_a", "situ_rot_quant_pack_a", "rot_quant_pack_a_i4", "pack_b", "gemm", "gemm_i4")

kernels = None
originals = {}
# ConvRot runs before the linear. rotate_activation leaves the rotation to the kernel by handing the tensor over here.
fusable = False
pending = None
# Set by linear_input_act when the next linear gets the raw [hidden | gate] input of a situ activation.
situ_next = False


def get_kernels():
    global kernels
    if kernels is None:
        kernels = Kernels("int8_gemm", KERNELS)
    return kernels


def run(quant, gemm, x, weight, weight_scale, bias, K):
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
    launch(quant, M, 1, 256, [c_ptr(x.data_ptr()), c_ptr(packed_a.data_ptr()), c_ptr(a_scale.data_ptr()), c_int(K)])
    launch(get_kernels().pack_b, row_bytes // 64, N // 64, 256, [c_ptr(weight.data_ptr()), c_ptr(packed_b.data_ptr()), c_int(row_bytes)])
    launch(gemm, blocks, N // 64, 256, [
        c_ptr(packed_a.data_ptr()), c_ptr(packed_b.data_ptr()), c_ptr(a_scale.data_ptr()), c_ptr(weight_scale.data_ptr()),
        c_ptr(bias.data_ptr() if bias is not None else None), c_ptr(out.data_ptr()), c_int(M), c_int(N), c_int(row_bytes // 64)])
    return out


def linear(x, weight, weight_scale, bias, rotate=False, situ=False):
    """x is (M, K) fp16, or (M, 2K) with situ, weight (N, K) int8 and weight_scale (N,) fp32. Returns (M, N) fp16."""
    k = get_kernels()
    quant = k.situ_rot_quant_pack_a if situ else k.rot_quant_pack_a if rotate else k.quant_pack_a
    return run(quant, k.gemm, x, weight, weight_scale, bias, weight.shape[1])


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
        global pending, situ_next
        rotate, situ = x is pending, situ_next
        pending, situ_next = None, False
        N, K = weight.shape
        if (x_q is not None or x.dtype != torch.float16 or compute_dtype != torch.float16 or (bias is not None and bias.dtype != torch.float16)
                or x.numel() // x.shape[-1] < MIN_ROWS or N % 64 != 0 or K % 64 != 0):
            if situ:
                x = originals["eager_input_act"](x, "situ")
            if rotate:
                x = originals["rotate_activation"](x, originals["build_hadamard"](ROT_GROUP, device=x.device, dtype=x.dtype), ROT_GROUP)
            return original(x, weight, weight_scale, bias, compute_dtype, x_q)

        if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() == N:
            scale = weight_scale.reshape(N).float().contiguous()
        else:
            scale = torch.full((N,), float(weight_scale), dtype=torch.float32, device=x.device)
        out = linear(x.reshape(-1, x.shape[-1]).contiguous(), weight.contiguous(), scale, bias, rotate, situ)
        return out.reshape(x.shape[:-1] + (N,))
    return int8_linear


def rotate_activation(x, H, group_size):
    global pending
    if situ_next:
        pending = x
        return x
    if fusable and group_size == ROT_GROUP and x.dtype == torch.float16 and x.dim() == 2 and x.shape[1] % ROT_GROUP == 0 and x.shape[1] <= MAX_ROT_K:
        pending = x
        return x
    return originals["rotate_activation"](x, H, group_size)


def fuse_situ(layer, x):
    """Only for layers where the kernel can do situ, ConvRot and the quantization in one pass."""
    return (getattr(layer, "_is_quantized", False) and getattr(layer, "_use_convrot", False) and not getattr(layer, "_is_w4a8", False)
            and getattr(layer, "_convrot_groupsize", 0) == ROT_GROUP and not layer.lora_patches and x.dtype == torch.float16
            and x.shape[-1] // 2 == layer.in_features and layer.in_features % ROT_GROUP == 0 and layer.in_features <= MAX_ROT_K)


def linear_input_act(linear, x, input_act, act_weight=None, act_eps=0.0, residual=None, residual_scale=None):
    global situ_next
    if input_act == "situ" and residual is None and fuse_situ(linear, x):
        situ_next = True
        return linear(x)
    return originals["linear_input_act"](linear, x, input_act, act_weight, act_eps, residual, residual_scale)


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
        originals.update(triton_int8_linear=pack.triton_int8_linear, triton_int8_linear_per_row=pack.triton_int8_linear_per_row,
                         rotate_activation=convrot.rotate_activation, build_hadamard=convrot.build_hadamard, forward=layer.forward,
                         linear_input_act=comfy.ops.linear_input_act, eager_input_act=comfy.ops._eager_input_act)

    for name in ("triton_int8_linear", "triton_int8_linear_per_row"):
        setattr(pack, name, wrap(originals[name]) if on else originals[name])
    convrot.rotate_activation = rotate_activation if on else originals["rotate_activation"]
    layer.forward = wrap_forward(originals["forward"]) if on else originals["forward"]
    comfy.ops.linear_input_act = linear_input_act if on else originals["linear_input_act"]
