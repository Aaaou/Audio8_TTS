"""Create U8U8 copies of the 0.1B U8S8 MatMulInteger graphs.

The conversion is mathematically lossless: 128 is added to each signed INT8
weight and to its zero point, leaving ``weight - zero_point`` unchanged. It
avoids the saturating U8S8 AVX2 accumulation path observed on an AMD Zen 3 CPU.
Original model files are preserved and remain selectable as ``int8``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[1]


def convert_graph(source: Path, destination: Path, force: bool) -> int:
    data_path = destination.parent / (destination.name + ".data")
    if destination.is_file() and data_path.is_file() and not force:
        onnx.checker.check_model(str(destination))
        print(f"Already present: {destination}")
        return 0
    model = onnx.load(str(source), load_external_data=True)
    initializers = {value.name: value for value in model.graph.initializer}
    replacements: dict[str, onnx.TensorProto] = {}
    converted: set[str] = set()
    for node in model.graph.node:
        if node.op_type != "MatMulInteger" or len(node.input) < 4:
            continue
        weight_name, zero_name = node.input[1], node.input[3]
        if weight_name not in initializers or zero_name not in initializers:
            raise RuntimeError(f"non-constant quantization values in {node.name}")
        if weight_name in converted:
            continue
        weight = numpy_helper.to_array(initializers[weight_name])
        zero = numpy_helper.to_array(initializers[zero_name])
        if weight.dtype != np.int8 or zero.dtype != np.int8:
            raise RuntimeError(f"expected INT8 weight and zero point in {node.name}")
        replacements[weight_name] = numpy_helper.from_array(
            (weight.astype(np.int16) + 128).astype(np.uint8), weight_name
        )
        replacements[zero_name] = numpy_helper.from_array(
            (zero.astype(np.int16) + 128).astype(np.uint8), zero_name
        )
        converted.add(weight_name)

    for index, value in enumerate(model.graph.initializer):
        if value.name in replacements:
            model.graph.initializer[index].CopyFrom(replacements[value.name])

    data_name = destination.name + ".data"
    if data_path.exists():
        data_path.unlink()
    onnx.save_model(
        model,
        str(destination),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_name,
        size_threshold=1024,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(destination))
    print(f"Converted {len(converted)} MatMulInteger weights: {destination}")
    return len(converted)


def update_manifest(model_dir: Path) -> None:
    path = model_dir / "runtime_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    available = list(manifest.get("available_precisions", []))
    if "int8" not in available:
        available.append("int8")
    if "u8u8" not in available:
        available.append("u8u8")
    manifest["available_precisions"] = available
    manifest.setdefault("slow_decode_models", {})["u8u8"] = "slow_ar_u8u8.onnx"
    manifest.setdefault("fast_models", {})["u8u8"] = "fast_ar_u8u8.onnx"
    manifest["default_precision"] = "u8u8"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated manifest; default precision is u8u8: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=ROOT / "model")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    required = [
        model_dir / "slow_ar_int8.onnx",
        model_dir / "fast_ar_int8.onnx",
        model_dir / "runtime_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("download the official model first; missing: " + ", ".join(missing))
    convert_graph(model_dir / "slow_ar_int8.onnx", model_dir / "slow_ar_u8u8.onnx", args.force)
    convert_graph(model_dir / "fast_ar_int8.onnx", model_dir / "fast_ar_u8u8.onnx", args.force)
    update_manifest(model_dir)


if __name__ == "__main__":
    main()
