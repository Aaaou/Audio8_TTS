from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from scripts.convert_u8s8_to_u8u8 import convert_graph


def test_u8u8_conversion_preserves_centered_weights(tmp_path: Path) -> None:
    source = tmp_path / "source.onnx"
    destination = tmp_path / "destination.onnx"
    weight = np.asarray([[-128, -3], [17, 127]], dtype=np.int8)
    zero = np.asarray(-4, dtype=np.int8)
    graph = helper.make_graph(
        [helper.make_node("MatMulInteger", ["a", "b", "a_zero", "b_zero"], ["y"])],
        "tiny_u8s8",
        [helper.make_tensor_value_info("a", TensorProto.UINT8, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.INT32, [1, 2])],
        [
            numpy_helper.from_array(weight, "b"),
            numpy_helper.from_array(np.asarray(5, dtype=np.uint8), "a_zero"),
            numpy_helper.from_array(zero, "b_zero"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save_model(model, source)

    assert convert_graph(source, destination, force=False) == 1
    converted = onnx.load(destination, load_external_data=True)
    values = {item.name: numpy_helper.to_array(item) for item in converted.graph.initializer}
    assert values["b"].dtype == np.uint8
    assert values["b_zero"].dtype == np.uint8
    np.testing.assert_array_equal(
        values["b"].astype(np.int16) - values["b_zero"].astype(np.int16),
        weight.astype(np.int16) - zero.astype(np.int16),
    )
