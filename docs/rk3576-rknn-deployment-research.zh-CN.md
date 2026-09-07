# Audio8 0.1B 在 RK3576 上的 RKNN 部署调研

## 目标

目标不是把整套 TTS 强制放入 NPU，而是将 Slow/Fast AR 的线性计算转换为
RKNN W8A8，允许状态更新、KV cache、Mamba 展开算子和 FP16 codec 继续由 CPU
处理。第一阶段以正确运行和数值对齐为准，性能优化放在板端验证之后。

## 工具链

- RKNN-Toolkit2 2.3.2；
- 目标平台 `rk3576`；
- Ubuntu 24.04 WSL2，Python 3.12；
- 板端使用与 Toolkit2 匹配的 RKNN Runtime/Lite2 2.3.2；
- C/C++ 部署使用 AArch64 Linux 交叉编译器。

RK3576 从 RKNN 2.0.0 起正式支持。Toolkit2 2.3.2 提供 CPython 3.6 至 3.12
的 x86-64 Linux wheel，以及对应的 AArch64 Lite2 wheel。转换在 x86-64 Linux
完成，`.rknn` 模型在开发板上运行；AArch64 交叉编译器不负责模型转换。

## 现有 ONNX 图检查

| 图 | 节点数 | `MatMulInteger` | `DynamicQuantizeLinear` | 普通 `MatMul` |
| --- | ---: | ---: | ---: | ---: |
| Slow AR U8U8 | 16,426 | 217 | 121 | 73 |
| Fast AR U8U8 | 1,116 | 21 | 17 | 8 |
| FP16 codec decoder | 3,341 | 0 | 0 | 60 |

RKNN 2.3.2 的官方算子表没有列出 `MatMulInteger` 和
`DynamicQuantizeLinear`，因此直接导入现有 U8S8/U8U8 图属于高风险路径。
`ScatterND`、`LayerNormalization`、普通 `MatMul` 等在 RKNN CPU OP 列表中；
RK3576 NPU 支持普通矩阵乘，但 shape、layout 和量化约束仍需编译器实测。

Codec 图还包含 `Einsum`、`Range` 和动态 frames。由于第一阶段让 codec 继续走
ONNX Runtime CPU，这些节点不阻塞 AR 的 RKNN 验证。

## 推荐转换路线

1. 使用官方 U8U8 图做结构模板，但不把其整数算子直接交给 RKNN。
2. 从官方原版 BF16 `model.safetensors` 读取 238 个线性层权重。
3. 将 `DynamicQuantizeLinear + MatMulInteger` 链恢复为普通浮点 `MatMul`。
4. 保持 Slow/Fast 输入输出、cache、conv state 和 SSM state 协议不变。
5. 使用真实 ONNX 推理状态制作多输入 `.npy` 校准集。
6. 用 RKNN-Toolkit2 对浮点图执行 W8A8 量化，目标设为 `rk3576`。
7. 先做 Fast AR 单步，再做 Slow AR 单步；通过后再接完整自回归循环。

U8U8 转换本身不使用数据集，因为它只是 S8 到 U8 的数学等价重编码。RKNN
W8A8 是一次新的激活量化，因此需要校准数据。数据不需要来自官方训练集；应从
原版模型和官方 ONNX Runtime 的真实推理循环捕获。

## 校准数据

Slow AR 每条样本包含：

```text
codes position cache_keys cache_values conv_states ssm_states
```

Fast AR 每条样本包含：

```text
slow_hidden token_id use_slow_hidden input_pos
cache_key_0 cache_value_0 ... cache_key_3 cache_value_3
```

RKNN 的 `dataset.txt` 每行代表一条样本，多输入文件按 ONNX 输入顺序用空格分隔。
所有输入保存为独立 `.npy` 文件。第一版建议覆盖 10 条中英文文本、prompt
prefill、首个生成 token、中间 token，以及 Fast AR 的 10 个 codebook 位置。

## 风险与判断门

| 风险 | 影响 | 判断方式 |
| --- | --- | --- |
| 现有整数 ONNX 算子不支持 | U8U8 不能直接导入 | `load_onnx` 探针 |
| 16,426 节点的 Slow 图编译失败 | 需要拆图或简化 | FP16 build 探针 |
| 大型显式 state 输入 | 内存和搬运开销 | 板端 RSS/CMA 与单步延迟 |
| CPU OP 与 NPU 频繁切换 | 能跑但更慢 | RKNN perf detail |
| W8A8 激活饱和 | token/音频错误 | logits、top-k、codes 对齐 |
| Runtime/驱动版本不匹配 | 板端初始化失败 | `rknn_server`/driver 版本检查 |

第一阶段成功标准：Fast 和 Slow 至少各有一个 `.rknn` 模型可构建，固定真实输入
可以在 simulator 或板端运行，输出无 NaN，首步 logits 与浮点基线高度相关。
完整音频与性能收益不是第一阶段的阻断条件。

## 当前结论

RK3576 平台本身受官方 Toolkit2 支持，且官方 Model Zoo 有 Whisper、Wav2Vec2、
Zipformer 和 MMS-TTS 示例。Audio8 的主要难点不在 TTS 类型，而在已经量化的
整数 ONNX 链、巨大的展开 Mamba 图和显式循环状态。最优路线是从 BF16 重建
普通浮点 AR 图，再由 RKNN 使用真实状态样本做 W8A8，而不是对 U8U8 文件进行
二次量化。
