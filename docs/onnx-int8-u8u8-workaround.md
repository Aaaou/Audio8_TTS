# Audio8 0.1B ONNX INT8 U8U8 workaround

This guide applies the mathematically equivalent U8S8-to-U8U8 workaround for
corrupted/noise-only output observed on an AMD AVX2 CPU. It does not requantize
from FP32 and does not require calibration data. Original model files remain
available as the `int8` precision.

## How the runnable files are replaced

The converter runs once and creates side-by-side files instead of overwriting
the official artifacts:

```text
slow_ar_int8.onnx(.data) -> slow_ar_u8u8.onnx(.data)
fast_ar_int8.onnx(.data) -> fast_ar_u8u8.onnx(.data)
```

It then updates `model/runtime_manifest.json` to register both variants and set
`default_precision` to `u8u8`. This branch's runtime follows that manifest, so
subsequent CLI and service launches load the new U8U8 graphs. Do not rename the
new files over the originals; keeping both variants makes rollback as simple as
passing `--precision int8`. The FP16 codec is unchanged.

For evidence and limitations, see the
[full bug report](onnx-int8-u8s8-avx2-bug-report.md).

## Fresh installation

Clone the fix branch:

```bash
git clone --branch fix/onnx-int8-u8s8-avx2 \
  https://github.com/Aaaou/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime_0_1b_int8
```

Windows PowerShell:

```powershell
py -3 -m pip install -U huggingface_hub
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
.\setup.ps1
.\.venv\Scripts\python.exe scripts\register_default_voice.py
```

Linux/macOS:

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
.venv/bin/python scripts/register_default_voice.py
```

The setup scripts automatically convert the graphs when the official model is
present. If setup runs before model download, it warns and skips conversion;
run it again after downloading.

## Repair an existing installation

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install "onnx>=1.16,<2"
.\.venv\Scripts\python.exe scripts\convert_u8s8_to_u8u8.py --model-dir model
```

Linux/macOS:

```bash
.venv/bin/python -m pip install 'onnx>=1.16,<2'
.venv/bin/python scripts/convert_u8s8_to_u8u8.py --model-dir model
```

Expected output includes:

```text
Converted 217 MatMulInteger weights: .../slow_ar_u8u8.onnx
Converted 21 MatMulInteger weights: .../fast_ar_u8u8.onnx
Updated manifest; default precision is u8u8
```

This creates about 170 MB of derived Slow/Fast graph files under the ignored
`model/` directory. The original U8S8 graphs and FP16 codec are not modified.

## Verify CLI inference

Windows:

```powershell
.\run_infer.ps1 `
  --precision u8u8 `
  --text "Hello, this is an Audio8 U8U8 test." `
  --voice default `
  --max-new-tokens 512 `
  --output outputs\u8u8-test.wav
```

Linux/macOS:

```bash
bash run_infer.sh \
  --precision u8u8 \
  --text "Hello, this is an Audio8 U8U8 test." \
  --voice default \
  --max-new-tokens 512 \
  --output outputs/u8u8-test.wav
```

## Verify the service

Windows:

```powershell
.\start_server.ps1
Invoke-RestMethod http://127.0.0.1:8024/api/health
```

Linux/macOS:

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
```

The health response should report:

```json
{
  "ok": true,
  "precision": "u8u8",
  "codec_precision": "fp16"
}
```

The service has switched only when the response reports `"precision":
"u8u8"`. If it still reports `int8`, restart the old process, rerun the
converter, and check that `ARKTTS_PRECISION` is not forcing the original graph.

Open `http://127.0.0.1:8024/`. The UI exposes a 3-70 second maximum duration
and estimates it from text length. `max_new_tokens` counts approximately 46.4ms
audio frames, not input characters.

## Select or roll back precision

CLI:

```text
--precision u8u8  # workaround
--precision int8  # original official graph
```

Service:

```bash
ARKTTS_PRECISION=u8u8 bash run_server.sh
ARKTTS_PRECISION=int8 bash run_server.sh
```

PowerShell uses the same `ARKTTS_PRECISION` environment variable. When it is
unset, the service follows `model/runtime_manifest.json`; successful conversion
sets its default to `u8u8`.

## Platform status

- Windows AMD Zen 3/AVX2: verified for the reported failure and workaround.
- Linux Intel/VNNI: original U8S8 verified; broader U8U8 benchmarking is pending.
- macOS Apple Silicon/x86-64: expected to work, but not hardware-tested.

If the health endpoint still reports `int8`, rerun the converter, restart the
old service process, check that `ARKTTS_PRECISION` is not forced to `int8`, and
hard-refresh the browser.
