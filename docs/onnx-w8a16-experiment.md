# Experimental 0.1B W8A16 ONNX conversion

This experiment applies the official 0.6B weight-only ONNX layout to the
0.1B model, changing only the packed linear weight precision from INT4 to
INT8. It is independent of the lossless U8U8 workaround.

## Quantization format

The generated Slow AR and Fast AR graphs use:

| Property | Official 0.6B INT4 | Experimental 0.1B W8A16 |
| --- | --- | --- |
| Operator | `com.microsoft::MatMulNBits` | `com.microsoft::MatMulNBits` |
| Weight bits | 4 | 8 |
| Activation inside linear op | FP16 | FP16 |
| Quantization | asymmetric, block-wise | asymmetric, block-wise |
| Block size | 128 | 128 |
| Accuracy level | 4 | 4 |
| Dynamic activation quantization | none | none |

The rest of the 0.1B graph remains FP32 because that is the public 0.1B ONNX
graph contract. Each converted linear operation casts its input to FP16,
runs `MatMulNBits(bits=8)`, and casts its result back to FP32. This avoids the
official 0.1B U8S8 `DynamicQuantizeLinear + MatMulInteger` path.

The converter reads all 238 linear weights directly from the original BF16
`Audio8-TTS-Preview-0.1b/model.safetensors`: 217 in Slow AR and 21 in Fast AR.
It does not dequantize or requantize the official W8A8 weights. The official
W8A8 graph is used only as a structural export template so that the input,
output, recurrent-state, and codec contracts remain unchanged.

## Build

Install the standalone runtime first and download both official model sets:

```bash
python3 -m pip install -r onnx_runtime_0_1b_int8/requirements.txt
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 \
  --local-dir onnx_runtime_0_1b_int8/model
hf download Audio8/Audio8-TTS-Preview-0.1b model.safetensors \
  --local-dir original_0_1b
python3 onnx_runtime_0_1b_int8/scripts/convert_to_w8a16.py \
  --model-dir onnx_runtime_0_1b_int8/model \
  --safetensors original_0_1b/model.safetensors
```

PowerShell:

```powershell
py -3 -m pip install -r onnx_runtime_0_1b_int8\requirements.txt
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 `
  --local-dir onnx_runtime_0_1b_int8\model
hf download Audio8/Audio8-TTS-Preview-0.1b model.safetensors `
  --local-dir original_0_1b
py -3 onnx_runtime_0_1b_int8\scripts\convert_to_w8a16.py `
  --model-dir onnx_runtime_0_1b_int8\model `
  --safetensors original_0_1b\model.safetensors
```

The command creates:

```text
slow_ar_w8a16.onnx
slow_ar_w8a16.onnx.data
fast_ar_w8a16.onnx
fast_ar_w8a16.onnx.data
```

It registers `w8a16` in `runtime_manifest.json` but deliberately does not make
it the default. Existing `int8` and `u8u8` files are preserved.

## Run and verify

```bash
python3 -m onnx_runtime_0_1b_int8.arktts_runtime.cli \
  --model-dir onnx_runtime_0_1b_int8/model \
  --voices-dir onnx_runtime_0_1b_int8/voices \
  --precision w8a16 \
  --text "这是语言合成量化效果测试" \
  --max-new-tokens 256 \
  --threads 1 \
  --output outputs/w8a16.wav
```

For the local service, set `ARKTTS_PRECISION=w8a16` before startup. Confirm
that `/api/health` reports `"precision": "w8a16"`.

Numerical validation should compare the first prefill logits before sampling,
not merely check that a WAV was written. Use:

```bash
python3 onnx_runtime_0_1b_int8/scripts/dump_cross_platform_diagnostics.py \
  --model-dir onnx_runtime_0_1b_int8/model \
  --voices-dir onnx_runtime_0_1b_int8/voices \
  --precision w8a16 \
  --text "这是语言合成量化效果测试" \
  --threads 1 \
  --output outputs/w8a16-diagnostics.json
```

## Current validation and limitations

On Windows 11, AMD Ryzen 5 5600, and ONNX Runtime 1.23.2:

- both graphs load with `CPUExecutionProvider`;
- all 238 linear operations are `MatMulNBits(bits=8)`;
- no `MatMulInteger` or `DynamicQuantizeLinear` node remains;
- an end-to-end WAV is generated successfully;
- first-step logits correlate at approximately 0.991 with the verified U8U8
  control and select the same first semantic token in the fixed diagnostic.

This is a real requantization from BF16 and is therefore not mathematically
identical to either BF16 or the U8U8 compatibility graph. Autoregressive token
sequences and waveforms can diverge after small logit changes. Listening tests,
longer prompts, Linux/macOS validation, real-time factor, peak memory, and
quality evaluation are still required before making W8A16 the default.

The first reproducible Windows/AMD performance comparison is documented in
[the W8A16 benchmark report](onnx-w8a16-benchmark-windows-amd.md). On that
system W8A16 is within about 2% of U8U8 performance and substantially faster
and smaller in peak RSS than the original PyTorch FP32 CPU path.

For memory-claim verification, the official U8S8 graphs were also loaded on
Ubuntu 22.04 x86-64 with ONNX Runtime 1.29.0. Slow AR, Fast AR, and the FP16
codec decoder used 633.95 MiB RSS at four threads and 635.62 MiB at eight
threads, consistent with the official “about 0.6 GB” figure. This is an
official-graph memory check, not a Linux W8A16 quality or speed validation.
