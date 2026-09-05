"""Build experimental 0.1B W8A16 MatMulNBits graphs.

This follows the official 0.6B ONNX INT4 linear-layer format, changing only
the packed weight precision from 4 to 8 bits:

* ``com.microsoft::MatMulNBits``
* asymmetric block-wise weights
* block size 128
* accuracy level 4
* FP16 activations inside each quantized linear operation

The source weights come from the original 0.1B BF16 ``model.safetensors``.
They are not recovered from, or requantized from, the official W8A8 graph.
The official W8A8 graph is used only as the structural ONNX export template.
Outside the converted linear operations, the 0.1B graph remains FP32.
"""
from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)


ROOT = Path(__file__).resolve().parents[1]


class SafetensorsReader:
    """Minimal read-only safetensors reader with BF16 support via NumPy."""

    _DTYPES = {
        "F64": np.dtype("<f8"),
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
        "BF16": np.dtype("<u2"),
        "I64": np.dtype("<i8"),
        "I32": np.dtype("<i4"),
        "I16": np.dtype("<i2"),
        "I8": np.dtype("i1"),
        "U64": np.dtype("<u8"),
        "U32": np.dtype("<u4"),
        "U16": np.dtype("<u2"),
        "U8": np.dtype("u1"),
        "BOOL": np.dtype("?"),
    }

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as stream:
            raw_header_size = stream.read(8)
            if len(raw_header_size) != 8:
                raise ValueError(f"invalid safetensors header: {path}")
            header_size = struct.unpack("<Q", raw_header_size)[0]
            self.header = json.loads(stream.read(header_size))
        self.data_offset = 8 + header_size

    def keys(self) -> set[str]:
        return {name for name in self.header if name != "__metadata__"}

    def tensor(self, name: str) -> np.ndarray:
        if name not in self.header or name == "__metadata__":
            raise KeyError(f"tensor not found in {self.path}: {name}")
        info = self.header[name]
        dtype_name = info["dtype"]
        if dtype_name not in self._DTYPES:
            raise ValueError(f"unsupported safetensors dtype {dtype_name!r}: {name}")
        start, end = info["data_offsets"]
        shape = tuple(info["shape"])
        dtype = self._DTYPES[dtype_name]
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if end - start != expected:
            raise ValueError(f"invalid tensor byte length: {name}")
        raw = np.memmap(
            self.path,
            mode="r",
            dtype=dtype,
            offset=self.data_offset + start,
            shape=shape,
            order="C",
        )
        if dtype_name == "BF16":
            # BF16 is the high 16 bits of IEEE float32.
            bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32) << 16
            return bits.view(np.float32)
        return np.array(raw, copy=True)


def state_key_for_node(node_name: str) -> str:
    suffix = "/MatMul_quant"
    if not node_name.startswith("/") or not node_name.endswith(suffix):
        raise ValueError(f"cannot map MatMulInteger node to a state key: {node_name}")
    return node_name[1 : -len(suffix)].replace("/", ".") + ".weight"


def _producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    return {output: node for node in model.graph.node for output in node.output}


def _consumer_map(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for value in node.input:
            consumers.setdefault(value, []).append(node)
    return consumers


def _only_consumer(
    consumers: dict[str, list[onnx.NodeProto]], value: str, op_type: str
) -> onnx.NodeProto:
    found = consumers.get(value, [])
    if len(found) != 1 or found[0].op_type != op_type:
        detail = [(node.name, node.op_type) for node in found]
        raise ValueError(f"expected one {op_type} consumer of {value}, got {detail}")
    return found[0]


def _prune_dead_nodes(model: onnx.ModelProto) -> None:
    producer = _producer_map(model)
    required_values = [value.name for value in model.graph.output]
    required_nodes: set[int] = set()
    while required_values:
        value = required_values.pop()
        node = producer.get(value)
        if node is None or id(node) in required_nodes:
            continue
        required_nodes.add(id(node))
        required_values.extend(node.input)
    kept = [node for node in model.graph.node if id(node) in required_nodes]
    del model.graph.node[:]
    model.graph.node.extend(kept)

    used = {value for node in kept for value in node.input}
    initializers = [value for value in model.graph.initializer if value.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)


def restore_weight_only_source(
    model: onnx.ModelProto, weights: SafetensorsReader
) -> list[str]:
    """Replace every W8A8 MatMulInteger chain with an FP16-weight MatMul.

    The returned node names are the only MatMul nodes passed to the NBits
    quantizer. Existing floating-point MatMul operators are deliberately left
    unchanged.
    """

    producer = _producer_map(model)
    consumers = _consumer_map(model)
    initializers = {value.name: value for value in model.graph.initializer}
    remove_ids: set[int] = set()
    replacements: dict[int, list[onnx.NodeProto]] = {}
    converted_names: list[str] = []
    new_initializers: list[onnx.TensorProto] = []

    matmul_integer_nodes = [node for node in model.graph.node if node.op_type == "MatMulInteger"]
    if not matmul_integer_nodes:
        raise ValueError("the structural template has no MatMulInteger nodes")

    for ordinal, node in enumerate(matmul_integer_nodes):
        if len(node.input) < 2 or node.input[1] not in initializers:
            raise ValueError(f"MatMulInteger has no constant weight: {node.name}")
        dynamic_quant = producer.get(node.input[0])
        if dynamic_quant is None or dynamic_quant.op_type != "DynamicQuantizeLinear":
            raise ValueError(f"MatMulInteger activation is not dynamically quantized: {node.name}")

        cast = _only_consumer(consumers, node.output[0], "Cast")
        output_mul = _only_consumer(consumers, cast.output[0], "Mul")
        other_input = next(value for value in output_mul.input if value != cast.output[0])
        scale_mul = producer.get(other_input)
        if scale_mul is None or scale_mul.op_type != "Mul":
            raise ValueError(f"missing activation/weight scale multiplication: {node.name}")
        if dynamic_quant.output[1] not in scale_mul.input:
            raise ValueError(f"dequantization scale does not use activation scale: {node.name}")

        state_key = state_key_for_node(node.name)
        source_weight = weights.tensor(state_key)
        template_shape = tuple(initializers[node.input[1]].dims)
        if source_weight.ndim != 2 or tuple(reversed(source_weight.shape)) != template_shape:
            raise ValueError(
                f"weight shape mismatch for {state_key}: state={source_weight.shape}, "
                f"ONNX={template_shape}"
            )

        # PyTorch Linear stores [out, in]; ONNX MatMul consumes [in, out].
        fp16_weight = np.ascontiguousarray(source_weight.T, dtype=np.float16)
        weight_name = f"w8a16.source_weight.{ordinal}"
        new_initializers.append(numpy_helper.from_array(fp16_weight, weight_name))

        base = node.name.removesuffix("_quant") + "_w8a16"
        activation_fp16 = base + ":activation_fp16"
        result_fp16 = base + ":result_fp16"
        matmul_name = base + "/MatMul"
        replacement = [
            helper.make_node(
                "Cast",
                [dynamic_quant.input[0]],
                [activation_fp16],
                name=base + "/CastIn",
                to=TensorProto.FLOAT16,
            ),
            helper.make_node(
                "MatMul",
                [activation_fp16, weight_name],
                [result_fp16],
                name=matmul_name,
            ),
            helper.make_node(
                "Cast",
                [result_fp16],
                [output_mul.output[0]],
                name=base + "/CastOut",
                to=TensorProto.FLOAT,
            ),
        ]
        remove_ids.update({id(node), id(cast), id(scale_mul), id(output_mul)})
        replacements[id(output_mul)] = replacement
        converted_names.append(matmul_name)

    rebuilt: list[onnx.NodeProto] = []
    for node in model.graph.node:
        if id(node) in replacements:
            rebuilt.extend(replacements[id(node)])
        elif id(node) not in remove_ids:
            rebuilt.append(node)
    del model.graph.node[:]
    model.graph.node.extend(rebuilt)
    model.graph.initializer.extend(new_initializers)
    _prune_dead_nodes(model)

    remaining = [node.name for node in model.graph.node if node.op_type == "MatMulInteger"]
    if remaining:
        raise ValueError(f"unconverted MatMulInteger nodes remain: {remaining[:5]}")
    return converted_names


def convert_graph(
    source: Path,
    destination: Path,
    weights: SafetensorsReader,
    force: bool,
) -> int:
    data_path = destination.parent / (destination.name + ".data")
    if destination.is_file() and data_path.is_file() and not force:
        onnx.checker.check_model(str(destination))
        print(f"Already present: {destination}")
        return 0

    print(f"Loading structural template: {source}")
    model = onnx.load(str(source), load_external_data=True)
    include_nodes = restore_weight_only_source(model, weights)
    print(f"Restored {len(include_nodes)} original BF16 linear weights")

    config = DefaultWeightOnlyQuantConfig(
        block_size=128,
        is_symmetric=False,
        accuracy_level=4,
        bits=8,
        op_types_to_quantize=("MatMul",),
    )
    quantizer = MatMulNBitsQuantizer(
        model,
        nodes_to_include=include_nodes,
        algo_config=config,
    )
    logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
    quantizer.process()
    quantized = quantizer.model.model

    nbits_nodes = [
        node
        for node in quantized.graph.node
        if node.domain == "com.microsoft" and node.op_type == "MatMulNBits"
    ]
    if len(nbits_nodes) != len(include_nodes):
        raise RuntimeError(
            f"expected {len(include_nodes)} MatMulNBits nodes, got {len(nbits_nodes)}"
        )
    for node in nbits_nodes:
        attributes = {item.name: helper.get_attribute_value(item) for item in node.attribute}
        expected = {"bits": 8, "block_size": 128, "accuracy_level": 4}
        if any(attributes.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"unexpected MatMulNBits attributes in {node.name}: {attributes}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if data_path.exists():
        data_path.unlink()
    quantizer.model.save_model_to_file(str(destination), use_external_data_format=True)
    onnx.checker.check_model(str(destination))
    print(f"Created {len(nbits_nodes)} W8A16 MatMulNBits nodes: {destination}")
    return len(nbits_nodes)


def update_manifest(model_dir: Path) -> None:
    path = model_dir / "runtime_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    available = list(manifest.get("available_precisions", []))
    if "w8a16" not in available:
        available.append("w8a16")
    manifest["available_precisions"] = available
    manifest.setdefault("slow_decode_models", {})["w8a16"] = "slow_ar_w8a16.onnx"
    manifest.setdefault("fast_models", {})["w8a16"] = "fast_ar_w8a16.onnx"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Registered experimental precision w8a16 (default unchanged): {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    parser.add_argument(
        "--safetensors",
        type=Path,
        required=True,
        help="Original Audio8 0.1B model.safetensors (BF16 source weights)",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    safetensors_path = args.safetensors.resolve()
    required = [
        model_dir / "slow_ar_int8.onnx",
        model_dir / "fast_ar_int8.onnx",
        model_dir / "runtime_manifest.json",
        safetensors_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing required input: " + ", ".join(missing))

    weights = SafetensorsReader(safetensors_path)
    convert_graph(
        model_dir / "slow_ar_int8.onnx",
        model_dir / "slow_ar_w8a16.onnx",
        weights,
        args.force,
    )
    convert_graph(
        model_dir / "fast_ar_int8.onnx",
        model_dir / "fast_ar_w8a16.onnx",
        weights,
        args.force,
    )
    update_manifest(model_dir)


if __name__ == "__main__":
    main()
