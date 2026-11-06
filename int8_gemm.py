import ctypes
import functools
import logging
import sys

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.ops
from comfy.ldm.krea2.model import SingleStreamBlock, SwiGLU

from . import hip_attention, rms_norm
from .hip_attention import Kernels, launch

BLOCK_ROWS = 512  # WAVES * 64 in int8_gemm.hip
ROT_THREADS = 128  # ROT_THREADS in int8_gemm.hip
MIN_ROWS = 512  # Below this the weight packing costs more than the kernel saves.
MAX_ROT_K = 16384  # Keeps the rotated row within the LDS of a block.
ROT_GROUP = 256

KERNELS = ("quant_pack_a", "rot_quant_pack_a", "situ_rot_quant_pack_a", "swiglu_rot_quant_pack_a", "gate_rot_quant_pack_a", "mod_rot_quant_pack_a", "rot_quant_pack_a_i4",
           "pack_b", "gemm", "gemm_i4", "gemm_k6144")

kernels = None
originals = {}
# ConvRot runs before the linear. rotate_activation leaves the rotation to the kernel by handing the tensor over here.
fusable = False
pending = None
# Set by linear_input_act when the next linear gets the raw [hidden | gate] input of a situ activation.
situ_next = False
# Set when the next linear gets a SwiGLU gate as its input, with this as the up projection.
swiglu_up = None
# Set when the next linear gets an attention output as its input, with this as the raw sigmoid gate.
sigmoid_gate = None
# (packed activations, scales) that the next linears read instead of quantizing their input, set while a block's linears share one input.
shared = None
# (x, gate) that the next linear adds to its output as x + gate * y.
residual = None
# The Int8 layer only reaches wrap through the Triton path of int8_quant.
swiglu_fusable = False


def get_kernels():
    global kernels
    if kernels is None:
        kernels = Kernels("int8_gemm", KERNELS)
    return kernels


def quantized(M, row_bytes, device):
    blocks = (M + BLOCK_ROWS - 1) // BLOCK_ROWS
    return torch.empty(blocks * BLOCK_ROWS * row_bytes // 4, dtype=torch.int32, device=device), torch.empty(M, dtype=torch.float32, device=device)


def run(quant, gemm, x, weight, weight_scale, bias, K, up=None, activations=None, res=None):
    """K is the number of values per row. weight is (N, row_bytes) int8 and weight_scale (N,) fp32. Returns (M, N) fp16."""
    M = x.shape[0]
    N, row_bytes = weight.shape
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)

    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    if activations is None:
        packed_a, a_scale = quantized(M, row_bytes, x.device)
        rotates = quant is not get_kernels().quant_pack_a
        inputs = [c_ptr(x.data_ptr())] if up is None else [c_ptr(x.data_ptr()), c_ptr(up.data_ptr())]
        launch(quant, M, 1, ROT_THREADS if rotates else 256, [*inputs, c_ptr(packed_a.data_ptr()), c_ptr(a_scale.data_ptr()), c_int(K)], K * 2 if rotates else 0)
    else:
        packed_a, a_scale = activations
    if gemm is get_kernels().gemm and row_bytes == 6144:
        gemm, b_ptr = get_kernels().gemm_k6144, weight.data_ptr()
    else:
        packed_b = torch.empty(N * row_bytes // 4, dtype=torch.int32, device=x.device)
        launch(get_kernels().pack_b, row_bytes // 64, N // 64, 256, [c_ptr(weight.data_ptr()), c_ptr(packed_b.data_ptr()), c_int(row_bytes)])
        b_ptr = packed_b.data_ptr()
    launch(gemm, (M + BLOCK_ROWS - 1) // BLOCK_ROWS, N // 64, 256, [
        c_ptr(packed_a.data_ptr()), c_ptr(b_ptr), c_ptr(a_scale.data_ptr()), c_ptr(weight_scale.data_ptr()),
        c_ptr(bias.data_ptr() if bias is not None else None), c_ptr(out.data_ptr()), c_int(M), c_int(N), c_int(row_bytes // 64),
        c_ptr(res[0].data_ptr() if res is not None else None), c_ptr(res[1].data_ptr() if res is not None else None), c_int(res[0].shape[1] if res is not None else 1)])
    return out


def linear(x, weight, weight_scale, bias, rotate=False, situ=False, up=None, gate=None, activations=None, res=None):
    """x is (M, K) fp16, or (M, 2K) with situ, weight (N, K) int8 and weight_scale (N,) fp32. With up, x is the gate of a SwiGLU. With gate, x is multiplied by its sigmoid. Returns (M, N) fp16."""
    k = get_kernels()
    quant = k.gate_rot_quant_pack_a if gate is not None else k.swiglu_rot_quant_pack_a if up is not None else k.situ_rot_quant_pack_a if situ else k.rot_quant_pack_a if rotate else k.quant_pack_a
    return run(quant, k.gemm, x, weight, weight_scale, bias, weight.shape[1], up if gate is None else gate, activations, res)


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
        global pending, situ_next, swiglu_up, sigmoid_gate, residual
        rotate, situ, up, gate, res = x is pending, situ_next, swiglu_up, sigmoid_gate, residual
        pending, situ_next, swiglu_up, sigmoid_gate, residual = None, False, None, None, None
        N, K = weight.shape
        if (x_q is not None or x.dtype != torch.float16 or compute_dtype != torch.float16 or (bias is not None and bias.dtype != torch.float16)
                or x.numel() // x.shape[-1] < MIN_ROWS or N % 64 != 0 or K % 64 != 0):
            if shared is not None or gate is not None or res is not None:
                raise RuntimeError("RDNA2 Attention: a fused Krea 2 block reached a layer the int8 kernels can't run.")
            if situ:
                x = originals["eager_input_act"](x, "situ")
            if up is not None:
                x = F.silu(x).mul_(up.reshape(x.shape))
            if rotate:
                x = originals["rotate_activation"](x, originals["build_hadamard"](ROT_GROUP, device=x.device, dtype=x.dtype), ROT_GROUP)
            return original(x, weight, weight_scale, bias, compute_dtype, x_q)

        if isinstance(weight_scale, torch.Tensor) and weight_scale.numel() == N:
            scale = weight_scale.reshape(N).float().contiguous()
        else:
            scale = torch.full((N,), float(weight_scale), dtype=torch.float32, device=x.device)
        out = linear(x.reshape(-1, x.shape[-1]).contiguous(), weight.contiguous(), scale, bias, rotate, situ, None if up is None else up.reshape(-1, K).contiguous(), gate, shared, res)
        return out.reshape(x.shape[:-1] + (N,))
    return int8_linear


def rotate_activation(x, H, group_size):
    global pending
    if shared is not None:
        return x
    if situ_next or swiglu_up is not None or sigmoid_gate is not None:
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


def quantize_block_input(x, norm, scale, shift):
    """Sets shared to the quantized (1 + scale) * norm(x) + shift, which is what all of a block's input linears read."""
    global shared
    B, L, K = x.shape
    packed_a, a_scale = quantized(B * L, K, x.device)
    weight = comfy.model_management.cast_to(norm.scale, dtype=torch.float32, device=x.device)
    scale, shift = (1 + scale).contiguous(), shift.contiguous()
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    launch(get_kernels().mod_rot_quant_pack_a, B * L, 1, ROT_THREADS, [c_ptr(x.data_ptr()), c_ptr(packed_a.data_ptr()), c_ptr(a_scale.data_ptr()), c_int(K), c_ptr(weight.data_ptr()),
                                                                       c_ptr(scale.data_ptr()), c_ptr(shift.data_ptr()), c_int(L), ctypes.c_float(norm.eps)], K * 2)
    shared = packed_a, a_scale


def block_forward(module, int8_qk, x, vec, freqs, mask=None, timestep_zero_index=None, transformer_options={}):
    """SingleStreamBlock.forward with everything between the matmuls fused. Rounds at the same points, so it matches fused_norm and fused_swiglu."""
    global shared, swiglu_up, sigmoid_gate, residual
    attn, mlp = module.attn, module.mlp
    patches = transformer_options.get("patches", {})
    layers = (attn.wq, attn.wk, attn.wv, attn.gate, attn.wo, mlp.gate, mlp.up, mlp.down)
    if (not swiglu_fusable or mask is not None or timestep_zero_index is not None or freqs is None
            or "attn1_patch" in patches or "attn1_output_patch" in patches or x.dtype != torch.float16 or x.shape[0] * x.shape[1] < MIN_ROWS or attn.headdim != rms_norm.HEAD_DIM
            or not all(isinstance(layer, originals["layer"]) and layer.bias is None and fuse_act(layer, x, layer.in_features) for layer in layers)):
        return SingleStreamBlock.forward(module, x, vec, freqs, mask, timestep_zero_index, transformer_options)

    B, L, D = x.shape
    prescale, preshift, pregate, postscale, postshift, postgate = module.mod(vec)
    try:
        quantize_block_input(x, module.prenorm, prescale, preshift)
        q, k, v, gate = attn.wq(x), attn.wk(x), attn.wv(x), attn.gate(x)
        shared = None

        q = rms_norm.rms_norm_rope(q, attn.heads, attn.qknorm.qnorm, freqs)
        k = rms_norm.rms_norm_rope(k, attn.kvheads, attn.qknorm.knorm, freqs)
        v = v.view(B, L, attn.kvheads, attn.headdim).transpose(1, 2)
        out = torch.empty((B, L, attn.heads, attn.headdim), dtype=x.dtype, device=x.device)
        hip_attention.attention(q, k, v, out.transpose(1, 2), attn.headdim ** -0.5, int8_qk)

        sigmoid_gate = gate.view(-1, D)
        residual = x, pregate.contiguous()
        x = attn.wo(out.view(B, L, D))

        quantize_block_input(x, module.postnorm, postscale, postshift)
        gate, up = mlp.gate(x), mlp.up(x)
        shared = None

        swiglu_up = up
        residual = x, postgate.contiguous()
        return mlp.down(gate)
    finally:
        # An interrupt must not leave these behind for the next layer that runs.
        shared = swiglu_up = sigmoid_gate = residual = None


def patch_block(model, int8_qk):
    for name, module in model.get_model_object("diffusion_model").named_modules():
        if isinstance(module, SingleStreamBlock):
            model.add_object_patch(f"diffusion_model.{name}.forward", functools.partial(block_forward, module, int8_qk))


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
