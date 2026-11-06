import ctypes
import functools

import comfy_kitchen
import torch
import torch.nn.functional as F

import comfy.ops
from comfy.ops import MixedPrecisionOp
from comfy.text_encoders.llama import MLP, BaseQwen3, FixedKV

from .hip_attention import Kernels, launch

BLOCK_ROWS = 16  # WAVES * ROWS in int8_gemv.hip
STEP = 512  # STEP in int8_gemv.hip
BF16_STEP = 256  # BF16_STEP in int8_gemv.hip
SPLITS = 32  # SPLITS in int8_gemv.hip
GROUPS = 4  # GROUPS in int8_gemv.hip
HEAD = 128  # HEAD in int8_gemv.hip
FAST_DEQUANT_GROUP = 256  # Larger ConvRot groups make kitchen's dequantize fall back to a dense matmul, 30x slower.

kernels = None
# Set inside generate(). Prompt encoding has to keep the exact rounding of core, because it decides the image.
generating = False


def get_kernels():
    global kernels
    if kernels is None:
        kernels = Kernels("int8_gemv", ("rotate", "gemv_i8", "gemv_bf16", "attn_decode", "attn_merge"))
    return kernels


def patched(module):
    return module.weight_function or module.bias_function or getattr(module, "weight_lowvram_function", None) is not None


def quantized_weight(module, weight_dtype, x):
    return comfy.ops.CastBiasWeightContext(module, input=None, dtype=weight_dtype, device=x.device, bias_dtype=x.dtype, offloadable=True, compute_dtype=x.dtype, want_requant=True)


def rotate(x, group):
    rotated = torch.empty_like(x)
    launch(get_kernels().rotate, x.numel() // group, 1, 256, [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(rotated.data_ptr()), ctypes.c_int(group)], group * 4)
    return rotated


def linear_forward(module, N, K, weight_dtype, group, x, *args, **kwargs):
    if x.dtype != torch.float32 or patched(module) or (x.numel() != K and (group <= FAST_DEQUANT_GROUP or not generating)):
        return type(module).forward(module, x, *args, **kwargs)

    k = get_kernels()
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    x = x.contiguous()
    with quantized_weight(module, weight_dtype, x) as (weight, bias):
        if group:
            x = rotate(x, group)
        if x.numel() != K:
            return F.linear(x, weight._qdata.to(x.dtype) * weight._params.scale.to(device=x.device), bias)
        out = torch.empty((*x.shape[:-1], N), dtype=torch.float32, device=x.device)
        scale = weight._params.scale.to(device=x.device)
        launch(k.gemv_i8, N // BLOCK_ROWS, 1, 256, [c_ptr(x.data_ptr()), c_ptr(weight._qdata.data_ptr()), c_ptr(scale.data_ptr()),
                                                    c_ptr(bias.data_ptr() if bias is not None else None), c_ptr(out.data_ptr()), c_int(K)], K * 4)
    return out


def mlp_forward(module, weight_dtype, x, norm=None):
    # rocBLAS is far from its best at N = 9728. One matmul at twice that is 2.8x faster than two, which is most of an fp32 prefill.
    if not generating or norm is not None or x.dtype != torch.float32 or x.numel() == x.shape[-1] or patched(module.gate_proj) or patched(module.up_proj):
        return type(module).forward(module, x, norm)

    with quantized_weight(module.gate_proj, weight_dtype, x) as (gate, _), quantized_weight(module.up_proj, weight_dtype, x) as (up, _):
        weight = torch.cat((gate.to(dtype=x.dtype).dequantize(), up.to(dtype=x.dtype).dequantize()))
    gate, up = F.linear(x, weight).chunk(2, dim=-1)
    return module.down_proj(module.activation(gate) * up)


def generate(transformer, *args, **kwargs):
    global generating
    generating = True
    try:
        return type(transformer).generate(transformer, *args, **kwargs)
    finally:
        generating = False


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


def attention_decode(q, k, v, lengths):
    """q is (batch, 1, heads, 128), k and v are (batch, capacity, kv heads, 128) and lengths (batch,) int32, all but lengths fp32."""
    batch, _, heads, _ = q.shape
    _, capacity, kv_heads, _ = k.shape
    kernels = get_kernels()
    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    q = q.contiguous()
    out = torch.empty_like(q)
    partial = torch.empty((batch * heads, SPLITS, HEAD), dtype=torch.float32, device=q.device)
    stats = torch.empty((batch * heads, SPLITS, 2), dtype=torch.float32, device=q.device)
    launch(kernels.attn_decode, SPLITS, batch * kv_heads, 256, [c_ptr(q.data_ptr()), c_ptr(k.data_ptr()), c_ptr(v.data_ptr()), c_ptr(lengths.data_ptr()),
                                                                c_ptr(partial.data_ptr()), c_ptr(stats.data_ptr()), c_int(capacity), c_int(kv_heads), ctypes.c_float(HEAD ** -0.5)])
    launch(kernels.attn_merge, batch * heads, 1, HEAD, [c_ptr(partial.data_ptr()), c_ptr(stats.data_ptr()), c_ptr(out.data_ptr())])
    return out


def init_kv_cache(model, batch, capacity, device, dtype):
    # A FixedKV cache is what lets core replay each layer from a graph. Kitchen only offers it with bf16, which this card doesn't have.
    config = model.config
    position = torch.empty((batch,), device=device, dtype=torch.int64)
    length = torch.zeros((batch,), device=device, dtype=torch.int32)
    caches = []
    for _ in range(config.num_hidden_layers):
        key = torch.empty((batch, capacity, config.num_key_value_heads, HEAD), device=device, dtype=dtype)
        caches.append(FixedKV(key, torch.empty_like(key), 0, position, length))
    return caches


def patch(clip):
    """Single token steps of text generation read int8 and bf16 weights directly instead of dequantizing them on every token."""
    for name, module in clip.patcher.model.named_modules():
        if isinstance(module, MixedPrecisionOp) and type(module).__name__ == "Linear" and module.quant_format == "int8_tensorwise":
            N, K = module.weight.shape
            params = module.weight._params
            if N % BLOCK_ROWS == 0 and K % STEP == 0 and params.scale.numel() == N and params.scale.dtype == torch.float32 and not params.transposed:
                group = params.convrot_groupsize if params.convrot else 0
                clip.patcher.add_object_patch(f"{name}.forward", functools.partial(linear_forward, module, N, K, module.weight.dtype, group))
        elif isinstance(module, MLP) and not module.merged_mlp and getattr(module.gate_proj, "quant_format", None) == "int8_tensorwise" and module.up_proj.quant_format == "int8_tensorwise":
            clip.patcher.add_object_patch(f"{name}.forward", functools.partial(mlp_forward, module, module.gate_proj.weight.dtype))
        elif isinstance(module, BaseQwen3) and not module.model.config.lm_head:
            embed = module.model.embed_tokens
            if embed.weight.dtype == torch.bfloat16 and embed.weight.shape[0] % BLOCK_ROWS == 0 and embed.weight.shape[1] % BF16_STEP == 0:
                clip.patcher.add_object_patch(f"{name}.logits", functools.partial(logits, module))
            clip.patcher.add_object_patch(f"{name}.generate", functools.partial(generate, module))
            config = module.model.config
            if module.model.fixed_kv and config.head_dim == HEAD and config.num_attention_heads == config.num_key_value_heads * GROUPS:
                comfy_kitchen.flash_attention_decode = attention_decode
                clip.patcher.add_object_patch(f"{name}.model.init_kv_cache", functools.partial(init_kv_cache, module.model))
