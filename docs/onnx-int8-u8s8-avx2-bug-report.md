# Audio8 0.1B ONNX INT8 U8S8 failure on an AMD AVX2 CPU

## Summary

The official `Audio8/audio8-TTS-0.1B-ONNX-INT8` graphs produced intelligible
speech on an Ubuntu Intel Xeon 8255C host with AVX-512 VNNI, but produced
noise-only/corrupted speech on Windows with an AMD Ryzen 5 5600 (Zen 3, AVX2).
The model files, voice codes, prompt, sampling settings, thread count, and ONNX
Runtime version were controlled. Divergence starts in the first call to
`slow_ar_int8.onnx`, before sampling, Fast AR, codec decoding, or WAV writing.

The Slow AR graph contains 217 `MatMulInteger` nodes and 121
`DynamicQuantizeLinear` nodes. Its main quantized path uses dynamic UINT8
activations and signed INT8 weights (U8S8). Re-encoding each signed weight and
its zero point as unsigned values by adding 128 preserves the centered integer
exactly while selecting a U8U8 path. On the affected AMD host, the converted
graph's first-step output matches the healthy Intel/VNNI output exactly and the
generated speech becomes intelligible.

This report calls the U8S8 AVX2 saturation/kernel explanation a strong diagnosis,
not a claim that every AMD processor or every ONNX Runtime build is affected.

## Environments

| Role | OS and CPU | ONNX Runtime | Result |
| --- | --- | --- | --- |
| Affected | Windows 11, AMD Ryzen 5 5600, AVX2 | 1.21.1, 1.22.1, 1.23.2 | Corrupted/noise-only speech |
| Healthy reference | Ubuntu, Intel Xeon Platinum 8255C, AVX-512 VNNI | 1.23.2 | Clear speech |

The affected machine produced identical bad tokens with one and four threads,
which excludes a thread race. The behavior persisted across the three tested
ORT versions, which makes a single-version regression unlikely.

## Controlled reproduction

```text
text: 这是语言合成量化效果测试，很清晰
voice: default
max_new_tokens: 64
temperature: 0.3
top_p: 0.9
top_k: 50
seed: 42
threads: 1
provider: CPUExecutionProvider
```

The constructed prompt was byte-identical:

```text
shape: [1, 11, 175]
SHA-256: d5232d57378b0dfdccfb390a846ee034be4672fe33c62172bd6c3ad8df043070
```

The default voice codes were also byte-identical. Relevant model hashes matched
on both hosts:

```text
slow_ar_int8.onnx
5aa5263d5c7e94cc58c81c969710045d313b4de87deae847f5388c126722bd3a

slow_ar_int8.onnx.data
4cf0f7ee7d5a9e817caaed310cc2f62dc08ebd37a1b87300dd4255ff0c45dace

fast_ar_int8.onnx
6cfc5c8e85d83d27d508d11351d8ca7d2749d6ff2990dc9efd048f86c27ff371

fast_ar_int8.onnx.data
cef092e658d8daad329117f9e9c3b09730deb3e39ea6f6634cd4d1ec8426eb9d

codec_decoder_fp16.onnx
25379b866ad555b9a55226c46325344d2ebfafea1b474c29de5223a9d01ea533

codec_decoder_fp16.onnx.data
b9bc968a84f41b86a9a3d19bc6c42b00b5de576512a87501d0c66ff7f896c954
```

## Stage isolation

The original U8S8 Slow AR output diverged at prefill position zero, before any
random sample was drawn:

```text
first-step logits correlation: 0.13422
first-step logits RMSE: 32.5052
```

The final 64-frame codec-token agreement was only 0.3125% for ORT 1.23.2 and
0.625% for ORT 1.22.1/1.21.1. Corresponding waveform correlations against the
Ubuntu reference were approximately zero.

To isolate the decoder, the healthy Ubuntu codec tokens were decoded with the
Windows FP16 codec graph:

```text
sample rate: 44100 / 44100
sample count: 131072 / 131072
waveform correlation: 0.9999999766
RMSE: 0.00001604
maximum absolute error: 0.00021362
```

This excludes the FP16 codec, SoundFile, WAV serialization, and playback path.

## Mathematically equivalent workaround

For every constant signed INT8 `MatMulInteger` weight `B` and weight zero point
`z`, the converter applies:

```text
B_u8 = uint8(int16(B_s8) + 128)
z_u8 = uint8(int16(z_s8) + 128)
```

Therefore:

```text
B_u8 - z_u8 == B_s8 - z_s8
```

No retraining or calibration is involved. The Slow graph converts 217 weights
and the Fast graph converts 21 weights. On the affected host:

```text
converted first-step logits exact match: true
converted first-step RMSE: 0
converted first-step correlation: 1.0
correlation after 175 recurrent prefill positions: 0.9986306
```

Small floating-point differences can accumulate in recurrent state and change a
later stochastic sample, so final WAV files are not expected to be byte-identical
across CPU families. The important checks are pre-sampling numerical agreement,
distributional stability, and intelligible output.

## Repository implementation

`onnx_runtime_0_1b_int8/scripts/convert_u8s8_to_u8u8.py` generates:

```text
model/slow_ar_u8u8.onnx
model/slow_ar_u8u8.onnx.data
model/fast_ar_u8u8.onnx
model/fast_ar_u8u8.onnx.data
```

It updates the downloaded `runtime_manifest.json`, registers both precisions,
and defaults to `u8u8`. Original U8S8 graphs remain available as `int8`. The
generated model directory is already ignored by Git, so approximately 170 MB of
derived weights are not committed to the source repository.

Both setup scripts perform the conversion when the official model is present.
If setup is run before model download, conversion is skipped with a warning and
can be rerun later. Service launchers no longer force `int8`; they allow the
downloaded manifest to select the successfully generated U8U8 variant. Explicit
fallbacks remain available:

```bash
bash run_infer.sh --precision int8 --text "test"
ARKTTS_PRECISION=int8 bash run_server.sh
```

## Platform compatibility and impact

| Platform | Validation level | Expected behavior |
| --- | --- | --- |
| Windows x86-64, AMD Zen 3/AVX2 | Fully validated for the reported case | U8U8 avoids the corrupted U8S8 path. |
| Linux x86-64, Intel AVX-512 VNNI | Original U8S8 validated | U8U8 is mathematically equivalent; broader performance validation remains useful. |
| macOS Apple Silicon/x86-64 | Not tested on hardware | Python conversion and ORT CPU EP are platform-neutral, but compatibility must not be presented as verified yet. |

Expected impact:

- about 170 MB additional disk space while retaining both graph variants;
- similar 8-bit model memory footprint during inference;
- CPU-dependent performance may differ between U8S8 and U8U8 kernels;
- setup requires the `onnx` Python package to rewrite and validate graphs;
- no changes to the FP16 codec or voice-code format;
- users can always select the original `int8` variant explicitly.

## Separate generation-length issue

`max_new_tokens` counts codec frames, not input characters. Each frame is 2048
samples at 44.1 kHz, approximately 46.4 ms:

```text
64 frames:    2.97 seconds
128 frames:   5.94 seconds
256 frames:  11.89 seconds
1024 frames: 47.55 seconds
1536 frames: 71.33 seconds
```

A fixed 64/128-frame UI limit cannot represent the documented guideline of up
to 150 input characters. The updated UI exposes a 3-70 second limit, estimates
it from Chinese-character and Latin-word counts, and converts seconds to frames.
The API and core runtime retain the current upstream 1024-frame default.

For the default UI sentence:

```text
你好，这是 Audio8 TTS 0.1B INT8 的本地语音合成测试。
```

a 12-second/259-frame ceiling produced 140 frames (6.5016 seconds) and stopped
naturally before the configured limit.

## Recommendations for maintainers

1. Publish an official U8U8 export, or test a U8S8 export using
   `reduce_range=True`.
2. Add Windows and Linux AMD AVX2 regression coverage in addition to VNNI hosts.
3. Store a first-step golden-logit/top-N-token test; a non-empty WAV is not a
   sufficient correctness check.
4. Retain an explicit precision override and document platform validation.
5. Express browser generation limits as time or estimate frames from text rather
   than confusing input characters with codec frames.

