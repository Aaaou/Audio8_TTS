# Audio8 0.1B ONNX INT8 U8U8 修复操作指南

本指南用于处理 `Audio8/audio8-TTS-0.1B-ONNX-INT8` 在部分 AVX2 CPU 上生成
噪声或“鬼叫”的问题。完整根因证据见
[U8S8 AVX2 Bug 报告](onnx-int8-u8s8-avx2-bug-report.zh-CN.md)。

该修复不会从 FP32 重新量化，也不需要校准数据。它将官方现有 S8 权重和零点同时
加 128，生成数学等价的 U8U8 ONNX 图，原始模型文件会保留。

## 实际替换方式

转换脚本只需要运行一次。它不会把新内容写回官方 `*_int8.onnx`，而是生成一套
并列文件：

```text
读取：slow_ar_int8.onnx + slow_ar_int8.onnx.data
生成：slow_ar_u8u8.onnx + slow_ar_u8u8.onnx.data

读取：fast_ar_int8.onnx + fast_ar_int8.onnx.data
生成：fast_ar_u8u8.onnx + fast_ar_u8u8.onnx.data
```

然后自动把 `model/runtime_manifest.json` 更新为等价于：

```json
{
  "default_precision": "u8u8",
  "available_precisions": ["int8", "u8u8"],
  "slow_decode_models": {
    "int8": "slow_ar_int8.onnx",
    "u8u8": "slow_ar_u8u8.onnx"
  },
  "fast_models": {
    "int8": "fast_ar_int8.onnx",
    "u8u8": "fast_ar_u8u8.onnx"
  }
}
```

本分支的 Runtime 根据 manifest 选择文件，所以实际运行时会用 `*_u8u8.onnx`。
这就是“替换可运行模型”的实现方式。不要手工把 U8U8 文件重命名成
`slow_ar_int8.onnx` 或覆盖原文件；并列保留可以用 `--precision int8` 立即回退。
FP16 codec 文件无需生成、替换或修改。

## 一、准备条件

- Python 3.10 或更高版本；
- 至少约 500MB 可用磁盘空间；
- 已下载官方 0.1B ONNX INT8 模型；
- Windows 使用 PowerShell；Linux/macOS 使用 Bash。

生成的 U8U8 Slow/Fast 图约额外占用 170MB。FP16 codec 不会复制或修改。

## 二、获取修复分支

```bash
git clone --branch fix/onnx-int8-u8s8-avx2 \
  https://github.com/Aaaou/Audio8_TTS.git
cd Audio8_TTS/onnx_runtime_0_1b_int8
```

已经克隆仓库时：

```bash
git fetch origin fix/onnx-int8-u8s8-avx2
git switch fix/onnx-int8-u8s8-avx2
cd onnx_runtime_0_1b_int8
```

## 三、全新安装

### Windows PowerShell

```powershell
py -3 -m pip install -U huggingface_hub
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
.\setup.ps1
.\.venv\Scripts\python.exe scripts\register_default_voice.py
```

如果 PowerShell 阻止脚本执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`setup.ps1` 会在安装依赖后自动执行 U8U8 转换。

### Linux

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
.venv/bin/python scripts/register_default_voice.py
```

### macOS

```bash
python3 -m pip install -U "huggingface_hub[cli]"
hf download Audio8/audio8-TTS-0.1B-ONNX-INT8 --local-dir model
bash setup.sh
.venv/bin/python scripts/register_default_voice.py
```

macOS 使用相同的 Python/ONNX 转换流程，但当前没有 Mac 实机测试，属于预期兼容，
不是已验证平台。

## 四、修复已有安装

不需要重新下载官方模型。先进入：

```text
Audio8_TTS/onnx_runtime_0_1b_int8
```

### Windows PowerShell

```powershell
.\.venv\Scripts\python.exe -m pip install "onnx>=1.16,<2"
.\.venv\Scripts\python.exe scripts\convert_u8s8_to_u8u8.py --model-dir model
```

### Linux/macOS

```bash
.venv/bin/python -m pip install 'onnx>=1.16,<2'
.venv/bin/python scripts/convert_u8s8_to_u8u8.py --model-dir model
```

成功时应看到类似输出：

```text
Converted 217 MatMulInteger weights: .../slow_ar_u8u8.onnx
Converted 21 MatMulInteger weights: .../fast_ar_u8u8.onnx
Updated manifest; default precision is u8u8: .../runtime_manifest.json
```

转换器会生成：

```text
model/slow_ar_u8u8.onnx
model/slow_ar_u8u8.onnx.data
model/fast_ar_u8u8.onnx
model/fast_ar_u8u8.onnx.data
```

以下官方文件不会被覆盖：

```text
model/slow_ar_int8.onnx
model/slow_ar_int8.onnx.data
model/fast_ar_int8.onnx
model/fast_ar_int8.onnx.data
model/codec_decoder_fp16.onnx
model/codec_decoder_fp16.onnx.data
```

需要重新生成时可增加 `--force`：

```powershell
.\.venv\Scripts\python.exe scripts\convert_u8s8_to_u8u8.py --model-dir model --force
```

## 五、验证命令行修复

### Windows PowerShell

```powershell
.\run_infer.ps1 `
  --precision u8u8 `
  --text "你好，这是 Audio8 TTS 0.1B INT8 的本地语音合成测试。" `
  --voice default `
  --max-new-tokens 512 `
  --threads 1 `
  --output outputs\u8u8-test.wav
```

### Linux/macOS

```bash
bash run_infer.sh \
  --precision u8u8 \
  --text "你好，这是 Audio8 TTS 0.1B INT8 的本地语音合成测试。" \
  --voice default \
  --max-new-tokens 512 \
  --threads 1 \
  --output outputs/u8u8-test.wav
```

命令应同时生成：

```text
outputs/u8u8-test.wav
outputs/u8u8-test.npy
```

## 六、启动和验证网页

### Windows

```powershell
.\start_server.ps1
Invoke-RestMethod http://127.0.0.1:8024/api/health
```

### Linux/macOS

```bash
bash start_server.sh
curl http://127.0.0.1:8024/api/health
```

健康接口应包含：

```json
{
  "ok": true,
  "precision": "u8u8",
  "codec_precision": "fp16"
}
```

只有看到 `"precision": "u8u8"` 才表示服务已经实际切换。如果仍显示 `int8`，
说明旧进程尚未重启、manifest 未更新，或环境变量强制指定了原版。

然后打开：

```text
http://127.0.0.1:8024/
```

网页提供“最长语音（秒）”，会按中英文文本长度自动估算，也可在 3–70 秒之间手动
调整。`max_new_tokens` 是约 46.4ms 一帧的音频帧数，不是输入字数。

## 七、强制选择精度

### 命令行

修复版：

```text
--precision u8u8
```

官方原始版：

```text
--precision int8
```

### 服务

Windows PowerShell 强制使用修复版：

```powershell
$env:ARKTTS_PRECISION = "u8u8"
.\run_server.ps1
```

Windows PowerShell 回退官方版：

```powershell
$env:ARKTTS_PRECISION = "int8"
.\run_server.ps1
```

Linux/macOS：

```bash
ARKTTS_PRECISION=u8u8 bash run_server.sh
# 或
ARKTTS_PRECISION=int8 bash run_server.sh
```

未设置环境变量时，服务读取 `model/runtime_manifest.json`；转换成功后默认是
`u8u8`。

## 八、常见问题

### 提示缺少官方模型

```text
FileNotFoundError: download the official model first
```

先执行 Hugging Face 下载命令，确认 `model/slow_ar_int8.onnx` 及其 `.data` 文件
存在，然后重新运行转换器。

### setup 提示跳过转换

说明运行 setup 时模型尚未下载。环境安装仍然有效；下载模型后重新执行 setup，或
按“修复已有安装”单独运行转换器。

### 健康接口仍显示 int8

1. 检查 `model/slow_ar_u8u8.onnx` 和 `.data` 是否存在；
2. 重新运行转换器；
3. 停止旧服务并重新启动；
4. 检查是否设置了 `ARKTTS_PRECISION=int8`；
5. 浏览器按 `Ctrl+F5` 刷新旧页面。

### 输出仍被截断

提高网页中的“最长语音（秒）”。换算参考：

```text
128 帧约 5.94 秒
256 帧约 11.89 秒
512 帧约 23.78 秒
1024 帧约 47.55 秒
1536 帧约 71.33 秒
```

### 是否可以删除原始 INT8 文件

不建议。保留它们便于对照和回退。U8U8 文件是由原始模型派生生成的。

## 九、恢复官方原始行为

无需删除 U8U8 文件，显式选择 `int8` 即可。如果需要完全清理派生文件，可删除四个
`*_u8u8.onnx*` 文件，再把 `runtime_manifest.json` 的默认 precision 改回 `int8`；
建议重新下载官方 manifest，避免手工编辑错误。
