# 0.1B W8A16 ONNX 实验方案

本实验把官方 0.6B 的 weight-only ONNX 量化方式应用到 0.1B，唯一的核心变化是
把线性层权重由 INT4 改成 INT8。它和数学无损的 U8U8 兼容修复是两套独立方案。

## 量化格式

| 属性 | 官方 0.6B INT4 | 实验版 0.1B W8A16 |
| --- | --- | --- |
| 算子 | `com.microsoft::MatMulNBits` | `com.microsoft::MatMulNBits` |
| 权重位宽 | 4 | 8 |
| 线性层内部激活 | FP16 | FP16 |
| 量化方式 | 非对称、分块 | 非对称、分块 |
| Block size | 128 | 128 |
| Accuracy level | 4 | 4 |
| 动态激活量化 | 无 | 无 |

0.1B 公开 ONNX 图的其余部分是 FP32，因此每个被转换的线性层先把输入转为
FP16，执行 `MatMulNBits(bits=8)`，然后把输出转回 FP32。Mamba、KV cache 和其余
算子的接口及精度不变，从而避开官方 0.1B 的 U8S8
`DynamicQuantizeLinear + MatMulInteger` 路径。

转换器直接读取原版 BF16 `Audio8-TTS-Preview-0.1b/model.safetensors`：Slow AR
217 个线性层、Fast AR 21 个线性层，共 238 个。它不会反量化或二次量化官方
W8A8 权重；官方 W8A8 图仅作为 ONNX 结构模板，以保留输入输出、循环状态和
codec 协议。

## 生成模型

先安装依赖，并下载官方 ONNX 模型与原版 BF16 权重：

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

Linux/macOS：

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

输出文件：

```text
slow_ar_w8a16.onnx
slow_ar_w8a16.onnx.data
fast_ar_w8a16.onnx
fast_ar_w8a16.onnx.data
```

转换器会在 `runtime_manifest.json` 中注册 `w8a16`，但不会把实验方案改成默认值，
也不会覆盖原始 `int8` 或现有 `u8u8` 文件。

## 推理与检查

```powershell
py -3 -m onnx_runtime_0_1b_int8.arktts_runtime.cli `
  --model-dir onnx_runtime_0_1b_int8\model `
  --voices-dir onnx_runtime_0_1b_int8\voices `
  --precision w8a16 `
  --text "这是语言合成量化效果测试" `
  --max-new-tokens 256 `
  --threads 1 `
  --output outputs\w8a16.wav
```

启动本地服务前设置 `ARKTTS_PRECISION=w8a16`；启动后检查 `/api/health` 是否返回
`"precision": "w8a16"`。

验证正确性时不能只看 WAV 是否成功写出，应在随机采样前比较首个 prefill logits：

```powershell
py -3 onnx_runtime_0_1b_int8\scripts\dump_cross_platform_diagnostics.py `
  --model-dir onnx_runtime_0_1b_int8\model `
  --voices-dir onnx_runtime_0_1b_int8\voices `
  --precision w8a16 `
  --text "这是语言合成量化效果测试" `
  --threads 1 `
  --output outputs\w8a16-diagnostics.json
```

## 当前验证结果和限制

在 Windows 11、AMD Ryzen 5 5600、ONNX Runtime 1.23.2 上已确认：

- Slow/Fast 两张图均可由 `CPUExecutionProvider` 加载；
- 238 个线性层全部变为 `MatMulNBits(bits=8)`；
- 图中不再存在 `MatMulInteger` 或 `DynamicQuantizeLinear`；
- 可以完成端到端推理并输出 WAV；
- 首步 logits 与已验证正常的 U8U8 对照相关系数约 0.991，固定诊断中选出的首个
  semantic token 相同。

这是从 BF16 重新量化得到的新模型，不与 BF16 或 U8U8 数学等价。自回归模型会把
很小的 logits 差异逐步放大，因此后续 token 和波形不必完全一致。将 W8A16 设为
默认方案前，还需要完成主观试听、长文本、Linux/macOS、实时系数、峰值内存和音质
评估。

首份可复现的 Windows/AMD 性能对比见
[W8A16 benchmark 报告](onnx-w8a16-benchmark-windows-amd.md)。在该机器上，
W8A16 与 U8U8 的性能差异约为 2%，同时明显快于原版 PyTorch FP32 CPU 路径，
峰值进程内存也更低。

另外在 Ubuntu 22.04 x86-64、ONNX Runtime 1.29.0 上加载官方 U8S8 Slow AR、
Fast AR 和 FP16 codec decoder：4 线程 RSS 为 633.95 MiB，8 线程为
635.62 MiB，与官方“约 0.6 GB”基本一致。该结果只复核官方图的加载内存，
不代表 W8A16 已在 Linux 上完成音质或速度验证。
