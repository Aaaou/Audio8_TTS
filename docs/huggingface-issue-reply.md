I reproduced the corrupted output of the official 0.1B ONNX INT8 model on
Windows with an AMD Ryzen 5 5600. The same files and settings produce clear
speech on an Intel Xeon/VNNI Ubuntu host. The divergence starts in the first
`slow_ar_int8.onnx` prefill call, before sampling or codec decoding (logits
correlation: 0.134). ORT 1.21.1, 1.22.1, and 1.23.2 all reproduce it, while
decoding the Ubuntu tokens on Windows matches the Ubuntu waveform with
0.99999998 correlation.

The graphs use dynamic U8 activations with S8 `MatMulInteger` weights. I made a
mathematically equivalent U8U8 copy by adding 128 to every S8 weight and its
zero point. On the affected AMD host, the first-step output then matches exactly
(RMSE 0, correlation 1.0), and speech becomes intelligible. This strongly points
to the U8S8 AVX2 accumulation/kernel path. Could you publish a U8U8 export or
test `reduce_range=True`, and add an AMD AVX2 regression test?

There is also a separate UI issue: `max_new_tokens` is an audio-frame count, not
an input-character count. 128 frames are only 5.94 seconds and cannot cover the
documented 150-character guideline. A text-based estimate or user-editable time
limit is safer than a fixed frame value.
