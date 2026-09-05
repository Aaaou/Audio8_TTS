# Audio8 0.1B INT8 ONNX Runtime

This directory is a self-contained runtime for the
`Audio8/audio8-TTS-0.1B-ONNX-INT8` export. It is intentionally separate from
the existing `onnx_runtime/` implementation, which targets the incompatible
0.6B INT4 graph.

## Why this runtime is separate

The 0.1B checkpoint uses a Falcon-H1 hybrid slow graph. The graph accepts one
`[1, 11, 1]` token column and carries four recurrent tensors between calls:

```text
cache_keys    [24, 1, 2, 2048, 64]
cache_values  [24, 1, 2, 2048, 64]
conv_states   [24, 1, 896, 4]
ssm_states    [24, 1, 24, 32, 64]
```

The implementation below performs prompt prefill one position at a time,
updates the returned state deltas, maps the compact 4097-way semantic logits,
then runs the four-layer Fast AR codebook graph and the FP16 codec decoder.

## Install

Python 3.10 or newer is supported, including the Python 3.10 environment used
by the Jetson issue report. From this directory:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
python scripts/register_default_voice.py
```

When the official model is already present, `setup.sh` also creates
mathematically equivalent U8U8 copies of the Slow/Fast AR graphs and registers
them as the default precision. The original U8S8 files are preserved. This
avoids corrupted output observed on an AMD AVX2 CPU while retaining an explicit
`--precision int8` fallback. Run the conversion directly when needed:

```bash
"$PWD/.venv/bin/python" scripts/convert_u8s8_to_u8u8.py --model-dir model
```

On Windows, use the native PowerShell entry points instead of `setup.sh`:

```powershell
py -3 -m pip install -U huggingface_hub
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
.\setup.ps1
.\.venv\Scripts\python.exe scripts\register_default_voice.py
```

`setup.ps1` performs the same optional U8U8 conversion after the dependencies
are installed. If setup is run before the model is downloaded, it prints a
warning and skips conversion; run it again after downloading, or execute:

```powershell
.\.venv\Scripts\python.exe scripts\convert_u8s8_to_u8u8.py --model-dir model
```

If PowerShell blocks local scripts, run them with
`powershell -ExecutionPolicy Bypass -File .\setup.ps1`.

The model directory must contain:

```text
model/
|- slow_ar_int8.onnx(.data)
|- fast_ar_int8.onnx(.data)
|- codec_decoder_fp16.onnx(.data)
|- runtime_manifest.json
|- reference_codes.npy
|- tokenizer/tokenizer.json
`- registration/
   |- codec_encoder_fp16.onnx(.data)        (optional)
   `- registration_manifest.json             (optional)
```

`register_default_voice.py` only uses `reference_codes.npy` and does not load
the optional encoder. To register a new voice from audio, use the service API
after the optional `registration/` files have been downloaded.

## CLI

```bash
bash run_infer.sh \
  --text "这是一个中文测试" \
  --voice default \
  --max-new-tokens 128 \
  --output outputs/test.wav
```

The command also writes `outputs/test.npy` with generated codes shaped
`[10, frames]`.

`max_new_tokens` counts audio codec frames, not input characters. Each frame is
2048 samples at 44.1 kHz (about 46.4 ms). The browser UI exposes a more intuitive
3-70 second limit and estimates it from text length. The documented 150-character
guideline may require roughly 800-1100 frames for slow Chinese speech; a fixed
128-frame limit is only about 5.94 seconds.

The equivalent Windows command is:

```powershell
.\run_infer.ps1 `
  --text "这是一个中文测试" `
  --voice default `
  --max-new-tokens 128 `
  --output outputs\test.wav
```

Set `ARKTTS_MODEL_DIR` or `ARKTTS_VOICES_DIR` to use another location. The
runtime always selects `CPUExecutionProvider`; GPU execution is outside the
scope of this export.

## Platform compatibility and precision selection

The compatibility conversion is based on model availability rather than the
operating-system name. After a successful conversion, the downloaded manifest
defaults to `u8u8` on Windows, Linux, and macOS. Override it at any time with
`--precision int8` for CLI use or `ARKTTS_PRECISION=int8` for the service.

| Platform | Status | Notes |
| --- | --- | --- |
| Windows x86-64, AMD Zen 3/AVX2 | Verified | Original U8S8 output was corrupted; U8U8 produces intelligible speech. |
| Linux x86-64, Intel AVX-512 VNNI | Original U8S8 verified | U8U8 is mathematically equivalent; cross-platform performance still needs broader benchmarking. |
| macOS Apple Silicon/x86-64 | Expected, not tested | Conversion and runtime contain no OS-specific kernel code, but no Mac hardware was available for validation. |

U8U8 remains 8-bit and should have similar model memory requirements, but
kernel performance can vary by CPU. Keeping both graph variants adds about
170 MB of disk use. The generated files live under the ignored `model/`
directory and are not committed to Git. See
[`docs/onnx-int8-u8s8-avx2-bug-report.md`](../docs/onnx-int8-u8s8-avx2-bug-report.md)
for evidence, limitations, and maintainer recommendations.
For step-by-step installation, existing-environment repair, verification, and
rollback, see the [U8U8 workaround guide](../docs/onnx-int8-u8u8-workaround.md).

## Local HTTP service

The service includes the same local browser UI as the 0.6B ONNX runtime. Open
`http://127.0.0.1:8024` after startup to enter text, select a voice, play or
download WAV output, register a reference voice, inspect memory, and reload
the runtime. The developer API remains available at `/docs`.

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
curl http://127.0.0.1:8024/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好，这是一个测试。","voice_name":"default"}' \
  -o outputs/api.wav
```

The service exposes `/api/tts`, `/api/tts/stream`, `/api/tts/cancel`,
`/api/voices`, `/api/voices/register`, `/api/registration/status`, and the
OpenAI-compatible `/v1/audio/speech`. Stop it with `bash stop_server.sh`.

On Windows PowerShell, use the managed wrappers:

```powershell
.\start_server.ps1
Invoke-WebRequest http://127.0.0.1:8024/api/health
.\stop_server.ps1
```

## Verification

```bash
"$PWD/.venv/bin/python" -m pytest -q tests
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

`test_contract.py` checks the exact input/output names and shapes when
`ARKTTS_MODEL_DIR` points at a downloaded model. Without model files it skips
the ONNX-dependent check; the pure prompt and state-shape tests still run.

The 0.1B model and this runtime are intended for local, authorized voice
generation. Obtain consent before cloning a voice and disclose synthetic audio
where appropriate.
