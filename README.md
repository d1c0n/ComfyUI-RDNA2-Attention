# ComfyUI-RDNA2-Attention

Faster attention and int8/int4 matmuls for ComfyUI on AMD RDNA2 (gfx1030).

RDNA2 has no matrix cores, and the usual fused kernels run poorly on it. This node replaces them with hand-written HIP kernels built around what the hardware does well.

Tested only on an RX 6900 XT, with ROCm 7.2 and PyTorch 2.14. Other cards are untested.

## Results

Seconds per image on an RX 6900 XT, 8 steps. Your numbers will vary.

| Model | Without this node | With this node |
|---|---|---|
| Anima, 832x1248 | 11.8s | 4.6-4.9s |
| Krea 2, int8 file, 856x856 | 38.5s | 13.2-13.8s |
| Krea 2, tuned int4 file, 856x856 | 26.7s (Triton int4) | 9.0-9.4s |
| MOSS-Image, int8, CFG 3, 704x1056 | 20.5s (normal loader) | 11.0-11.6s |

## Requirements

- An RDNA2 GPU with gfx1030: RX 6800, 6800 XT, 6900 XT or 6950 XT. The kernels are compiled for gfx1030 only.
- PyTorch with ROCm.
- `hipcc` on your `PATH`. Kernels compile on first use, which takes about a second.
- [ComfyUI-INT8-Fast-ROCM](https://github.com/patientx/ComfyUI-INT8-Fast-ROCM), only if you want `int8_gemm`.

## Install

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/d1c0n/ComfyUI-RDNA2-Attention.git
```

Restart ComfyUI. Launch it normally, no extra flag is needed for the node itself.

## Usage

Add **RDNA2 Attention** (under `model/patch`) after your model loader and any LoRAs, right before the sampler:

```
Model loader -> LoRAs -> RDNA2 Attention -> KSampler
```

| Option | What it does |
|---|---|
| `backend` | `hip` is the fused attention kernel, and the best default. `hip_int8` uses int8 for `QK`, which is about 5% faster overall and slightly less exact. `bmm` uses rocBLAS matmuls. |
| `int8_gemm` | Runs models from the Int8 and Int4 loaders on faster kernels. Needs ComfyUI-INT8-Fast-ROCM. |
| `fused_norm` | Krea 2 only. Runs RMSNorm as one fp16 kernel instead of casting to fp32 and back, which is 7-10% faster overall. Accuracy is the same, but the same seed can give a different image. |
| `fused_swiglu` | Krea 2 int8 files only, needs `int8_gemm`. Folds the SwiGLU activation into the quantization of the `down` layer, about 2% faster overall. Images do not change. |
| `transpose_weights` | Experimental, leave it off. Not tested on real workflows, and it is skipped under dynamic VRAM. |

Anything the fast kernels cannot handle falls back automatically. `hip` needs fp16 and head size 128, without masks, so SDXL uses `bmm`. `int8_gemm` needs at least 512 rows. `fused_norm` needs fp16. `fused_swiglu` skips int4 layers, unquantized layers and layers with Dynamic LoRA patches.

### Text generation

Add **RDNA2 Text Generate** after the CLIP loader and connect it to both Generate Text and CLIP Text Encode:

```
CLIP loader -> RDNA2 Text Generate -> Generate Text, CLIP Text Encode
```

ComfyUI otherwise dequantizes every int8 weight on every generated token on this card. The node reads the weights directly on single token steps and adds the fp32 decode attention that lets ComfyUI replay each layer from a graph, about 20x faster per token with the same output. Graphs need dynamic VRAM, without them it is about 8x. Prompt encoding does not change. It needs an int8 text encoder for the linear layers. Layers with LoRA patches fall back.

## Recommended setup per model

Tick `int8_gemm` for the models that use the Int8 or Int4 loader. Set `weight_dtype` to fp16. For Krea 2, also tick `fused_norm`, plus `fused_swiglu` on the int8 file.

| Model | Loader | `model_type` | On-the-fly | ConvRot |
|---|---|---|---|---|
| Anima | Load Diffusion Model Int8 | anima | On | On |
| Krea 2, int8 file | Load Diffusion Model Int8 | krea2 | Off | On |
| Krea 2, tuned int4 file | Load Diffusion Model Int4 | krea2 | Off | Built in |
| MOSS-Image, int8 file | Load Diffusion Model Int8 | anima (no MOSS option, unused here) | Off | On |
| SDXL / Illustrious | Your normal loader, `int8_gemm` off | | | |

## Important notes

**Quality**

- Images will not match other backends exactly on the same seed. These models amplify tiny numeric changes, so details shift.

**LoRAs**

- On the Int8 loader, use `lora_mode` **Stochastic**. The default, None, visibly broke the MOSS DMD LoRA. Stochastic stays clean at full speed.
- LoRAs on int4 are untested. None and Stochastic re-round patched layers, which may discard tuning. Dynamic avoids that, but those layers fall back to the slower Triton path.

**Side effects to know about**

- `int8_gemm` is global. The last RDNA2 Attention node to run sets it for every Int8 and Int4 model, including workflows without this node.
- This node patches functions inside ComfyUI-INT8-Fast-ROCM at runtime, so a change in that pack can break `int8_gemm`.
- Installing this node makes MOSS-Image compute in fp16 instead of fp32, about twice as fast, even in workflows without the node. MOSS does not officially list fp16, so check your results. To undo it, remove the `supported_inference_dtypes` line in `__init__.py`.

**Other**

- The Int4 loader's on-the-fly mode crashes in the version tested, and it does not convert int8 files to int4.
- `prototypes/` holds the original benchmark code and is not used by the node.

## How it works

Tiled kernels on RDNA2 stall near 50% of peak because they run out of LDS bandwidth. These kernels avoid LDS instead:

- Each lane owns its own rows, kept in registers.
- The other operand is identical across the wave, so it streams through the scalar cache and is read directly as an SGPR operand of the dot instructions, at full rate.
- Waves in a block warm the scalar cache together, so misses overlap instead of stalling one at a time.

The int8 and int4 matmuls use `v_dot4c_i32_i8` and `v_dot8_i32_i4` this way, reaching about 65 and 100-120 TOPS. ConvRot, quantization and the `situ` or SwiGLU activation are fused into one pass over the activations.

## License

GPL-3.0. See [LICENSE](LICENSE).
