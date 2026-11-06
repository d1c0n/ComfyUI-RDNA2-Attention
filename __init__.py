import logging

import torch
from typing_extensions import override

import comfy.supported_models
from comfy.patcher_extension import CallbacksMP
from comfy.quant_ops import QuantizedTensor
from comfy_api.latest import ComfyExtension, io

from . import int8_gemm as int8_kernels
from . import int8_gemv
from . import rms_norm
from .attention import attention_bmm, attention_hip, attention_hip_int8

BACKENDS = {"hip": attention_hip, "hip_int8": attention_hip_int8, "bmm": attention_bmm}

# bf16 is slow on RDNA2, so without fp16 here ComfyUI computes MOSS in fp32, which is about 2x slower.
comfy.supported_models.MOSSImage.supported_inference_dtypes = [torch.float16, torch.bfloat16, torch.float32]


def relayout_weights(patcher):
    # rocBLAS is about 10% faster with K-major weights. The shape stays the same, so F.linear picks that path on its own.
    for module in patcher.model.diffusion_model.modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        w = module.weight
        if isinstance(w, QuantizedTensor) or w.dtype != torch.float16 or w.device.type == "cpu" or min(w.shape) < 1024 or w.stride(0) == 1:
            continue
        w.data = w.data.t().contiguous().t()


class RDNA2Attention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RDNA2Attention",
            display_name="RDNA2 Attention",
            category="model/patch",
            description="Faster attention for RX 6900 XT class GPUs (gfx1030). Put it after the model loader and any LoRAs.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input("backend", options=list(BACKENDS), default="hip",
                               tooltip="hip: fused kernel for fp16 and head dim 128, needs hipcc, anything else uses bmm. hip_int8: same with int8 QK, a little faster and less exact. bmm: rocBLAS matmuls."),
                io.Boolean.Input("transpose_weights", default=False,
                                 tooltip="Store fp16 weights K-major for about 1.09x faster steps. Needs --highvram or --disable-dynamic-vram, it is skipped with dynamic VRAM."),
                io.Boolean.Input("int8_gemm", default=False,
                                 tooltip="Run Load Diffusion Model Int8 layers on the fast scalar-fed kernel. Stays active for every workflow until a node turns it off."),
                io.Boolean.Input("fused_norm", default=False,
                                 tooltip="Krea 2 only: run RMSNorm as one fp16 kernel instead of casting to fp32 and back. About 1 value in 12000 changes by one fp16 step, which can still change the image."),
                io.Boolean.Input("fused_swiglu", default=False,
                                 tooltip="Krea 2 Int8 only, needs int8_gemm: fold the SwiGLU activation into the quantization of the down layer. Other layers run as before."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, backend, transpose_weights=False, int8_gemm=False, fused_norm=False, fused_swiglu=False) -> io.NodeOutput:
        m = model.clone()
        int8_kernels.enable(int8_gemm)
        m.set_model_optimized_attention(BACKENDS[backend])
        if fused_norm:
            rms_norm.patch(m)
        if fused_swiglu:
            int8_kernels.patch_swiglu(m)
        if transpose_weights:
            # Dynamic VRAM rebuilds weights from flat contiguous buffers, so it can't keep this layout.
            if m.is_dynamic():
                logging.warning("RDNA2 Attention: transpose_weights skipped, it needs --highvram or --disable-dynamic-vram.")
            else:
                m.add_callback(CallbacksMP.ON_PRE_RUN, relayout_weights)
        return io.NodeOutput(m)


class RDNA2TextGenerate(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="RDNA2TextGenerate",
            display_name="RDNA2 Text Generate",
            category="model/patch",
            description="Much faster Generate Text with int8 text encoders on RX 6900 XT class GPUs (gfx1030). Only single token steps change, so prompt encoding is unaffected.",
            inputs=[io.Clip.Input("clip")],
            outputs=[io.Clip.Output()],
        )

    @classmethod
    def execute(cls, clip) -> io.NodeOutput:
        clip = clip.clone()
        int8_gemv.patch(clip)
        return io.NodeOutput(clip)


class RDNA2AttentionExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [RDNA2Attention, RDNA2TextGenerate]


async def comfy_entrypoint() -> RDNA2AttentionExtension:
    return RDNA2AttentionExtension()
