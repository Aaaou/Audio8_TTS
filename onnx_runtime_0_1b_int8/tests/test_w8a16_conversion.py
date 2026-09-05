from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from scripts.convert_to_w8a16 import (
    SafetensorsReader,
    restore_weight_only_source,
    state_key_for_node,
)


def _write_bf16_safetensors(path: Path, name: str, value: np.ndarray) -> None:
    value = np.asarray(value, dtype=np.float32)
    bf16 = (value.view(np.uint32) >> 16).astype("<u2")
    payload = bf16.tobytes()
    header = json.dumps(
        {
            name: {
                "dtype": "BF16",
                "shape": list(value.shape),
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_safetensors_reader_decodes_bf16(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    expected = np.asarray([[1.5, -2.25], [0.125, 9.0]], dtype=np.float32)
    _write_bf16_safetensors(path, "linear.weight", expected)
    actual = SafetensorsReader(path).tensor("linear.weight")
    np.testing.assert_array_equal(actual, expected)


def test_state_key_mapping() -> None:
    assert (
        state_key_for_node("/slow/layers.3/mamba/in_proj/MatMul_quant")
        == "slow.layers.3.mamba.in_proj.weight"
    )
    assert (
        state_key_for_node("/fast_layers.0/attention/wqkv/MatMul_quant")
        == "fast_layers.0.attention.wqkv.weight"
    )


def test_restore_uses_original_weight_and_fp16_matmul() -> None:
    dql = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["x_q", "x_scale", "x_zero"],
        name="quantize_x",
    )
    integer = helper.make_node(
        "MatMulInteger",
        ["x_q", "weight_q", "x_zero", "weight_zero"],
        ["integer_result"],
        name="/slow/layers.0/mamba/in_proj/MatMul_quant",
    )
    cast = helper.make_node("Cast", ["integer_result"], ["cast_result"], to=TensorProto.FLOAT)
    scale_mul = helper.make_node("Mul", ["x_scale", "weight_scale"], ["combined_scale"])
    output_mul = helper.make_node("Mul", ["cast_result", "combined_scale"], ["y"])
    graph = helper.make_graph(
        [dql, integer, cast, scale_mul, output_mul],
        "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
        [
            numpy_helper.from_array(np.zeros((2, 3), dtype=np.int8), "weight_q"),
            numpy_helper.from_array(np.zeros((3,), dtype=np.int8), "weight_zero"),
            numpy_helper.from_array(np.ones((3,), dtype=np.float32), "weight_scale"),
        ],
    )
    model = helper.make_model(graph)

    original = np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

    class FakeWeights:
        @staticmethod
        def tensor(name: str) -> np.ndarray:
            assert name == "slow.layers.0.mamba.in_proj.weight"
            return original

    include = restore_weight_only_source(model, FakeWeights())  # type: ignore[arg-type]
    assert include == ["/slow/layers.0/mamba/in_proj/MatMul_w8a16/MatMul"]
    assert not [node for node in model.graph.node if node.op_type == "MatMulInteger"]
    assert not [node for node in model.graph.node if node.op_type == "DynamicQuantizeLinear"]
    matmul = next(node for node in model.graph.node if node.op_type == "MatMul")
    weight = next(value for value in model.graph.initializer if value.name == matmul.input[1])
    restored = numpy_helper.to_array(weight)
    assert restored.dtype == np.float16
    np.testing.assert_array_equal(restored, original.T.astype(np.float16))
    casts = [node for node in model.graph.node if node.op_type == "Cast"]
    assert len(casts) == 2
