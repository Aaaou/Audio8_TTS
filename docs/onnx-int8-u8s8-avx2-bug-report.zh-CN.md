# Audio8 0.1B ONNX INT8 在 AMD AVX2 CPU 上的 U8S8 异常报告

## 摘要

官方 `Audio8/audio8-TTS-0.1B-ONNX-INT8` 在带 AVX-512 VNNI 的 Ubuntu
Intel Xeon 8255C 上可以生成清晰语音，但同一套模型、输入和 ONNX Runtime 在
Windows AMD Ryzen 5 5600（Zen 3、AVX2）上生成无意义噪声。差异从
`slow_ar_int8.onnx` 的第一次调用就开始，早于随机采样、Fast AR、codec 解码和
WAV 写入。

Slow AR 图包含 217 个 `MatMulInteger` 和 121 个 `DynamicQuantizeLinear`，主要
量化路径是动态 UINT8 激活乘有符号 INT8 权重（U8S8）。将每个 S8 权重和对应零点
同时加 128 后，中心化整数值不变，但图会走 U8U8 路径。在受影响的 AMD 主机上，
转换后首步输出与健康 Intel/VNNI 对照逐元素相同，语音恢复为可辨识中文。

当前证据强烈指向 U8S8 AVX2 的饱和累计或相关 kernel 数值问题，但不主张所有 AMD
CPU 或所有 ONNX Runtime 构建都必然受影响。

## 测试环境

| 角色 | 系统和 CPU | ONNX Runtime | 结果 |
| --- | --- | --- | --- |
| 异常环境 | Windows 11、AMD Ryzen 5 5600、AVX2 | 1.21.1、1.22.1、1.23.2 | 噪声/“鬼叫” |
| 正常对照 | Ubuntu、Intel Xeon Platinum 8255C、AVX-512 VNNI | 1.23.2 | 清晰语音 |

Windows 单线程与四线程输出逐字节相同，可排除线程竞争；三个 ORT 版本均复现，可
排除单一版本回归。

## 严格复现条件

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

两端构造出的 prompt 完全相同：

```text
shape: [1, 11, 175]
SHA-256: d5232d57378b0dfdccfb390a846ee034be4672fe33c62172bd6c3ad8df043070
```

默认音色 codes 也完全相同。关键模型文件哈希：

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

因此可排除下载不完整和错误权重。

## 分阶段定位

原始 U8S8 Slow AR 在 prefill 位置 0、尚未进行随机采样时就已经分叉：

```text
首步 logits 相关系数：0.13422
首步 logits RMSE：32.5052
```

最终 64 帧 codec token 一致率：ORT 1.23.2 仅为 0.3125%，ORT 1.22.1 和
1.21.1 均为 0.625%；对应 WAV 与 Ubuntu 对照的相关系数接近零。

固定 Ubuntu 正常 codec tokens 后，在 Windows 使用相同 FP16 codec 解码：

```text
采样率：44100 / 44100
采样点：131072 / 131072
波形相关系数：0.9999999766
RMSE：0.00001604
最大绝对误差：0.00021362
```

因此可排除 FP16 codec、SoundFile、WAV 序列化和播放链路。

## 数学等价修复

对每个 `MatMulInteger` 的常量 S8 权重 `B` 和权重零点 `z`：

```text
B_u8 = uint8(int16(B_s8) + 128)
z_u8 = uint8(int16(z_s8) + 128)
```

恒有：

```text
B_u8 - z_u8 == B_s8 - z_s8
```

该转换不涉及重新训练或重新校准。Slow 图转换 217 个权重，Fast 图转换 21 个权重。
在受影响主机上：

```text
首步逐元素相同：是
首步 RMSE：0
首步相关系数：1.0
175 个递归 prefill 位置后的相关系数：0.9986306
```

递归状态会放大不同 CPU 的微小浮点舍入差异，后续随机采样可能不同，因此不要求不同
CPU 最终 WAV 逐字节相同；关键标准是采样前数值一致性、分布稳定性和语音可懂度。

## 仓库实现

`onnx_runtime_0_1b_int8/scripts/convert_u8s8_to_u8u8.py` 生成：

```text
model/slow_ar_u8u8.onnx
model/slow_ar_u8u8.onnx.data
model/fast_ar_u8u8.onnx
model/fast_ar_u8u8.onnx.data
```

脚本更新下载目录中的 `runtime_manifest.json`，注册 `int8`/`u8u8` 并默认选择
`u8u8`。原始 U8S8 文件始终保留。生成文件位于仓库已经忽略的 `model/` 目录，约
170MB 的派生权重不会提交到 GitHub。

Windows PowerShell 和 POSIX setup 脚本都会在检测到官方模型后执行转换；如果先安装
环境、后下载模型，会给出警告并跳过，下载完成后重新执行 setup 即可。服务启动脚本
不再强制写死 `int8`，而是读取转换后 manifest。仍可显式回退：

```bash
bash run_infer.sh --precision int8 --text "test"
ARKTTS_PRECISION=int8 bash run_server.sh
```

## 跨平台兼容性和影响

| 平台 | 验证级别 | 预期行为 |
| --- | --- | --- |
| Windows x86-64、AMD Zen 3/AVX2 | 已完整验证当前案例 | U8U8 避开异常 U8S8 路径。 |
| Linux x86-64、Intel AVX-512 VNNI | 原始 U8S8 已验证 | U8U8 数学等价，仍建议扩大性能测试。 |
| macOS Apple Silicon/x86-64 | 未实机验证 | Python 转换和 ORT CPU EP 不依赖 OS，但不能宣称已验证。 |

影响：

- 同时保留两套图约增加 170MB 磁盘；
- 推理时仍是 8 位，模型运行内存理论上接近原 INT8；
- 不同 CPU kernel 的速度可能有差异；
- setup 增加 `onnx` 依赖，用于改写并校验模型；
- FP16 codec 和音色 codes 格式不变；
- 用户可随时显式选择原始 `int8`。

## 独立的生成长度问题

`max_new_tokens` 是 codec 帧，不是输入字数。每帧为 44.1kHz 下的 2048 个采样点，
约 46.4ms：

```text
64 帧：    2.97 秒
128 帧：   5.94 秒
256 帧：  11.89 秒
1024 帧： 47.55 秒
1536 帧： 71.33 秒
```

固定 64/128 帧无法覆盖官方“150 个字以内”的输入建议。更新后的网页提供 3–70 秒
上限，按中文字数和英文单词数估算，并将秒数转换为帧。API 和核心 Runtime 保留最新
上游的 1024 帧默认值。

默认网页句子在 12 秒/259 帧上限下实际生成 140 帧（6.5016 秒），并在撞到上限前
自然停止。

## 给维护者的建议

1. 发布官方 U8U8 导出，或验证使用 `reduce_range=True` 的 U8S8 导出。
2. 除 VNNI 主机外，加入 Windows/Linux AMD AVX2 回归测试。
3. 保存首步 golden logits 或 top-N tokens；“生成了非空 WAV”不能证明正确。
4. 保留显式 precision 回退，并准确标注各平台验证级别。
5. 网页用时间表达生成上限或按文本估算帧数，不要混淆输入字数和 codec 帧。
