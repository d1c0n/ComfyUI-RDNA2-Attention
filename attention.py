import torch

from comfy.ldm.modules.attention import attention_pytorch

from . import hip_attention

# Keeps each chunk of attention scores inside the 128MB Infinity Cache of RDNA2 GPUs.
BMM_CHUNK_BYTES = 128 * 1024 * 1024


def to_heads(t, b, heads, dim_head):
    return t.reshape(b, -1, heads, dim_head).transpose(1, 2)


def attention_bmm(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
    if mask is not None or kwargs.get("enable_gqa", False):
        return attention_pytorch(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    b = q.shape[0]
    dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if not skip_reshape:
        q, k, v = (to_heads(t, b, heads, dim_head) for t in (q, k, v))

    batch, q_len, k_len = b * heads, q.shape[2], k.shape[2]
    # rocBLAS is about 2x faster on the NN layout, and transposing from a contiguous k is about 8x faster than from the strided one.
    q = q.reshape(batch, q_len, dim_head).contiguous()
    k = k.reshape(batch, k_len, dim_head).contiguous().transpose(1, 2).contiguous()
    v = v.reshape(batch, k_len, -1)

    out = torch.empty((batch, q_len, v.shape[-1]), dtype=q.dtype, device=q.device)
    step = min(batch, max(1, BMM_CHUNK_BYTES // (q_len * k_len * q.element_size())))
    scores = torch.empty((step, q_len, k_len), dtype=q.dtype, device=q.device)
    scale = kwargs.get("scale", dim_head ** -0.5)
    for i in range(0, batch, step):
        s = scores[:min(step, batch - i)]
        torch.baddbmm(s, q[i:i + step], k[i:i + step], beta=0, alpha=scale, out=s)
        torch.softmax(s, -1, out=s)
        torch.bmm(s, v[i:i + step], out=out[i:i + step])

    out = out.view(b, heads, q_len, -1)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(b, q_len, -1)
    return out


def attention_hip(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, int8_qk=False, **kwargs):
    dim_head = q.shape[-1] if skip_reshape else q.shape[-1] // heads
    if mask is not None or kwargs.get("enable_gqa", False) or dim_head != 128 or v.shape[-1] != q.shape[-1] or q.dtype != torch.float16:
        return attention_bmm(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

    b = q.shape[0]
    if not skip_reshape:
        q, k, v = (to_heads(t, b, heads, dim_head) for t in (q, k, v))

    q_len = q.shape[2]
    q, k, v = (t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v))
    out = torch.empty((b, q_len, heads, dim_head), dtype=q.dtype, device=q.device).transpose(1, 2)
    hip_attention.attention(q, k, v, out, kwargs.get("scale", dim_head ** -0.5), int8_qk)
    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, q_len, heads * dim_head)


def attention_hip_int8(*args, **kwargs):
    return attention_hip(*args, int8_qk=True, **kwargs)
