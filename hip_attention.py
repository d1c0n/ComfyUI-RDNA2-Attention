import ctypes
import math
import os
import subprocess
import tempfile

import torch

BLOCK_ROWS = 256  # WAVES * 32 in hip_attention.hip

hip = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so"))


def check(error):
    if error != 0:
        raise RuntimeError(f"HIP error {error}")


def compile_kernels(name):
    source = os.path.join(os.path.dirname(__file__), name + ".hip")
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, name + ".hsaco")
        subprocess.run(["hipcc", "--genco", "-O3", "--offload-arch=gfx1030", source, "-o", out], check=True)
        with open(out, "rb") as f:
            return f.read()


class Kernels:
    def __init__(self, name, functions):
        self.code = compile_kernels(name)
        self.module = ctypes.c_void_p()
        check(hip.hipModuleLoadData(ctypes.byref(self.module), self.code))
        for function in functions:
            func = ctypes.c_void_p()
            check(hip.hipModuleGetFunction(ctypes.byref(func), self.module, function.encode()))
            setattr(self, function, func)


def launch(func, grid_x, grid_y, block, args):
    pointers = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in args])
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    check(hip.hipModuleLaunchKernel(func, grid_x, grid_y, 1, block, 1, 1, 0, stream, pointers, None))


kernels = None


def attention(q, k, v, out, scale, int8_qk=False):
    """q, k, v, out are (batch, heads, length, 128) fp16 tensors with a contiguous last dimension."""
    global kernels
    if kernels is None:
        kernels = Kernels("hip_attention", ("pack_kv", "attn", "pack_kv_i8", "attn_i8"))

    b, heads, q_len, _ = q.shape
    k_len = k.shape[2]
    groups = (k_len + 15) // 16
    # The extra group is what the last iteration of attn prefetches.
    packed = torch.empty(b * heads * (groups + 1) * (1552 if int8_qk else 2048), dtype=torch.int32, device=q.device)

    c_int, c_ptr = ctypes.c_int, ctypes.c_void_p
    launch(kernels.pack_kv_i8 if int8_qk else kernels.pack_kv, groups, b * heads, 256, [
        c_ptr(k.data_ptr()), c_ptr(v.data_ptr()), c_ptr(packed.data_ptr()), c_int(k_len), c_int(groups), c_int(heads),
        *[c_int(s) for s in k.stride()[:3]], *[c_int(s) for s in v.stride()[:3]]])
    launch(kernels.attn_i8 if int8_qk else kernels.attn, (q_len + BLOCK_ROWS - 1) // BLOCK_ROWS, b * heads, BLOCK_ROWS, [
        c_ptr(q.data_ptr()), c_ptr(packed.data_ptr()), c_ptr(out.data_ptr()), c_int(q_len), c_int(k_len), c_int(groups), c_int(heads),
        *[c_int(s) for s in q.stride()[:3]], *[c_int(s) for s in out.stride()[:3]], ctypes.c_float(scale * math.log2(math.e))])
