# Audio8 0.1B INT8 ONNX Runtime

这是 `Audio8/audio8-TTS-0.1B-ONNX-INT8` 的独立 CPU ONNX Runtime。它放在
单独目录中，因为仓库现有的 `onnx_runtime/` 是给 0.6B INT4 图使用的，不能
直接处理 0.1B Falcon-H1 hybrid 图。

0.1B Slow AR 每次只接受一个 `[1, 11, 1]` token column，并且要在调用之间保存：

```text
cache_keys    [24, 1, 2, 2048, 64]
cache_values  [24, 1, 2, 2048, 64]
conv_states   [24, 1, 896, 4]
ssm_states    [24, 1, 24, 32, 64]
```

本实现会逐位置执行 prompt prefill，写回 Slow AR 返回的 state delta，处理
4097 维 compact semantic logits，然后调用四层 Fast AR 和 FP16 codec decoder。

## 安装

支持 Python 3.10 及以上版本，Jetson issue 中的 Python 3.10 也可以使用：

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
python scripts/register_default_voice.py
```

如果官方模型已经下载，`setup.sh` 还会生成数学等价的 U8U8 Slow/Fast AR 图并将
其注册为默认精度。原始 U8S8 文件会保留，可用 `--precision int8` 显式回退。
这用于规避在 AMD AVX2 CPU 上确认的异常噪声输出。也可以单独执行：

```bash
"$PWD/.venv/bin/python" scripts/convert_u8s8_to_u8u8.py --model-dir model
```

Windows 用户请使用 PowerShell 脚本，不需要运行 `setup.sh`：

```powershell
py -3 -m pip install -U huggingface_hub
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
.\setup.ps1
.\.venv\Scripts\python.exe scripts\register_default_voice.py
```

`setup.ps1` 会执行同样的可选 U8U8 转换。如果安装环境时模型还没下载，脚本会警告
并跳过；下载模型后重新运行 setup，或执行：

```powershell
.\.venv\Scripts\python.exe scripts\convert_u8s8_to_u8u8.py --model-dir model
```

如果 PowerShell 禁止执行本地脚本，可以运行：
`powershell -ExecutionPolicy Bypass -File .\setup.ps1`。

模型文件放在本目录的 `model/` 下。`registration/` 下的 encoder 是可选的，
只在从音频注册新音色时需要。默认音色脚本只读取 `reference_codes.npy`。

## 命令行推理

```bash
bash run_infer.sh \
  --text "这是一个中文测试" \
  --voice default \
  --max-new-tokens 128 \
  --output outputs/test.wav
```

同时会生成 `[10, frames]` 形状的 `outputs/test.npy`。模型或音色放在其他位置时，
可设置 `ARKTTS_MODEL_DIR`、`ARKTTS_VOICES_DIR`。

`max_new_tokens` 表示音频 codec 帧，不是输入字数。每帧为 44.1 kHz 下的 2048 个
采样点，约 46.4ms；128 帧只有约 5.94 秒。网页改为显示 3–70 秒的最长语音，并
按照中英文文本长度自动估算。接近官方建议上限的 150 个字，慢速中文可能需要约
800–1100 帧。

## 跨平台兼容性和精度选择

转换成功后，下载目录的 manifest 会在 Windows、Linux 和 macOS 上统一默认使用
`u8u8`，而不是根据操作系统名称猜测 CPU kernel。命令行可用 `--precision int8`，
服务可用 `ARKTTS_PRECISION=int8` 随时回退官方原始图。

| 平台 | 状态 | 说明 |
| --- | --- | --- |
| Windows x86-64，AMD Zen 3/AVX2 | 已验证 | 原始 U8S8 输出异常；U8U8 恢复可辨识语音。 |
| Linux x86-64，Intel AVX-512 VNNI | 原始 U8S8 已验证 | U8U8 数学等价，但仍需更广泛的跨 CPU 性能测试。 |
| macOS Apple Silicon/x86-64 | 预期兼容，未实测 | 转换和 Runtime 没有 OS 专用 kernel 代码，但当前没有 Mac 实机验证。 |

U8U8 仍为 8 位，模型运行内存理论上接近原 INT8，但不同 CPU kernel 的速度可能
不同。同时保留两组 Slow/Fast 图会增加约 170MB 磁盘占用。生成文件位于已忽略的
`model/`，不会提交到 Git。完整证据、限制和上游建议见
[`docs/onnx-int8-u8s8-avx2-bug-report.zh-CN.md`](../docs/onnx-int8-u8s8-avx2-bug-report.zh-CN.md)。
全新安装、已有环境修复、验证和回退步骤见
[U8U8 修复操作指南](../docs/onnx-int8-u8u8-workaround.zh-CN.md)。
另有一套独立实验方案：从原版 BF16 权重生成与 0.6B 相同路径的 weight-only
`MatMulNBits(bits=8)` 图。具体命令和限制见
[W8A16 实验指南](../docs/onnx-w8a16-experiment.zh-CN.md)；目前不会默认启用。

Windows PowerShell 推理命令：

```powershell
.\run_infer.ps1 `
  --text "这是一个中文测试" `
  --voice default `
  --max-new-tokens 128 `
  --output outputs\test.wav
```

## HTTP 服务

服务同时提供与 0.6B ONNX Runtime 相同的本地网页界面。启动后打开
`http://127.0.0.1:8024`，即可输入文本、选择音色、播放或下载 WAV、注册参考音色、
查看内存状态和重新加载运行时；开发者 API 文档仍在 `/docs`。

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
```

服务提供 `/api/tts`、`/api/tts/stream`、`/api/tts/cancel`、音色查询和注册接口，
以及兼容 OpenAI 的 `/v1/audio/speech`。使用 `bash stop_server.sh` 停止服务。

Windows PowerShell 使用：

```powershell
.\start_server.ps1
Invoke-WebRequest http://127.0.0.1:8024/api/health
.\stop_server.ps1
```

## 测试

```bash
"$PWD/.venv/bin/python" -m pytest -q tests
```

Windows：

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests
```

请在获得授权后进行音色克隆，并在适当场景披露合成音频。
