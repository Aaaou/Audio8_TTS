# Audio8 0.1B CPU benchmark: FP32, U8U8, and W8A16

Date: 2026-09-06

## Test system

- Windows 11
- AMD Ryzen 5 5600, 6 physical cores / 12 logical processors
- 32 GiB system memory
- Python 3.12.14
- ONNX Runtime 1.23.2, CPUExecutionProvider
- PyTorch 2.14.0+cpu
- Four inference threads and one inter-op thread

The original PyTorch model runs in FP32 on CPU. Its optional Mamba and
causal-convolution fast paths are unavailable on this system, so the official
naive implementation is measured.

## Method

All backends use the same:

- text: `这是语言合成量化效果测试`;
- precomputed default reference codes and reference transcript;
- temperature 0.7, top-p 0.9, top-k 50, seed 42;
- maximum 256 generated codec frames;
- one warm-up followed by three measured runs.

Every backend is measured in a separate process. Model load time ends only
after the AR model and codec are both loaded. Peak RSS covers loading, warm-up,
generation, and decoding. Reported timings are the median of three runs.

The three models naturally stop at slightly different lengths because real
BF16-to-INT8 quantization changes logits and autoregressive sampling. For this
reason, AR throughput in frames/s and end-to-end real-time factor are more
comparable than raw wall-clock time.

## Results

| Metric | PyTorch FP32 | ONNX U8U8 | ONNX W8A16 |
| --- | ---: | ---: | ---: |
| Generated frames | 62 | 60 | 63 |
| Audio duration | 2.879 s | 2.786 s | 2.926 s |
| Ready-to-run load time | 2.858 s | 3.453 s | 3.329 s |
| AR generation time | 8.338 s | 5.992 s | 6.389 s |
| Codec decode time | 1.935 s | 2.157 s | 2.298 s |
| End-to-end time | 10.346 s | 8.119 s | 8.687 s |
| AR throughput | 7.436 frames/s | 10.014 frames/s | 9.860 frames/s |
| End-to-end RTF | 3.593 | 2.914 | 2.969 |
| Peak process RSS | 3673 MiB | 1579 MiB | 1573 MiB |
| Dual-AR files on disk | 323.87 MiB* | 167.39 MiB | 168.92 MiB |

`*` The PyTorch number is the complete BF16 safetensors checkpoint, whereas
the ONNX number is the Slow/Fast AR graph pair. Codec files are excluded from
all disk figures because their formats differ and they are not changed by the
W8A16 experiment.

## Interpretation

Compared with U8U8, W8A16 is effectively tied on this CPU:

- W8A16 AR throughput is 1.53% lower;
- W8A16 RTF is 1.91% higher (slower);
- measured peak RSS differs by less than 6 MiB;
- W8A16 Dual-AR files use approximately 1.53 MiB more disk space.

Compared with the original PyTorch FP32 CPU path:

- W8A16 AR throughput is 32.6% higher;
- W8A16 end-to-end RTF is 17.4% lower (faster per second of audio);
- W8A16 measured peak process RSS is 57.2% lower;
- the W8A16 Dual-AR files are approximately 47.8% smaller than the BF16
  checkpoint.

U8U8 remains slightly faster and is mathematically equivalent to the official
W8A8 model. W8A16's benefit is architectural: it avoids dynamic activation
quantization and the problematic U8S8 `MatMulInteger` path while retaining
FP16 activation inside each quantized linear layer. Performance alone does not
currently justify replacing U8U8 with W8A16 as the default.

These results apply to this CPU, operating system, ORT version, prompt, and
thread setting. Longer utterances and other AVX2/VNNI/Apple Silicon CPUs should
be benchmarked independently.

## Reproduce

Use `onnx_runtime_0_1b_int8/scripts/benchmark_backends.py`. Run each command in
a fresh process. Example for W8A16:

```powershell
py -3 onnx_runtime_0_1b_int8\scripts\benchmark_backends.py `
  --backend onnx `
  --model-dir onnx_runtime_0_1b_int8\model `
  --voices-dir onnx_runtime_0_1b_int8\voices `
  --precision w8a16 `
  --threads 4 `
  --repeats 3 `
  --max-new-tokens 256 `
  --output benchmark_w8a16.json
```
