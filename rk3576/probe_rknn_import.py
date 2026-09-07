"""Probe RKNN import/build support before committing to full calibration."""
from __future__ import annotations

import argparse
from pathlib import Path

from rknn.api import RKNN


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    output = args.output or args.model.with_suffix(".rk3576.rknn")
    rknn = RKNN(verbose=True)
    try:
        ret = rknn.config(target_platform="rk3576")
        if ret != 0:
            raise RuntimeError(f"rknn.config failed: {ret}")
        ret = rknn.load_onnx(model=str(args.model))
        if ret != 0:
            raise RuntimeError(f"rknn.load_onnx failed: {ret}")
        ret = rknn.build(
            do_quantization=args.quantize,
            dataset=str(args.dataset) if args.dataset else None,
        )
        if ret != 0:
            raise RuntimeError(f"rknn.build failed: {ret}")
        ret = rknn.export_rknn(str(output))
        if ret != 0:
            raise RuntimeError(f"rknn.export_rknn failed: {ret}")
        print(f"exported: {output}")
    finally:
        rknn.release()


if __name__ == "__main__":
    main()
