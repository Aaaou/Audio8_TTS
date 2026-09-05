"""Measure RSS after each ONNX Runtime loading stage.

This intentionally separates model-load RSS from generation peak RSS. Run in a
fresh process for each precision and keep the resulting JSON with the benchmark.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arktts_runtime.runtime import _session


def rss_mib(process: psutil.Process) -> float:
    return process.memory_info().rss / 1024**2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--precision", default="u8u8")
    parser.add_argument("--codec-precision", default="fp16")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    manifest = json.loads((model_dir / "runtime_manifest.json").read_text(encoding="utf-8"))
    slow_name = manifest.get("slow_decode_models", {}).get(args.precision)
    fast_name = manifest.get("fast_models", {}).get(args.precision)
    codec_name = manifest.get("codec_models", {}).get(
        args.codec_precision, f"codec_decoder_{args.codec_precision}.onnx"
    )
    if not slow_name or not fast_name:
        raise ValueError(f"precision {args.precision!r} is not present in runtime_manifest.json")

    process = psutil.Process(os.getpid())
    stages: list[dict[str, float | str]] = []

    def record(name: str) -> None:
        gc.collect()
        stages.append({"stage": name, "rss_mib": round(rss_mib(process), 2)})

    record("python_baseline")
    slow = _session(model_dir / slow_name, args.threads)
    record("slow_session_loaded")
    fast = _session(model_dir / fast_name, args.threads)
    record("slow_fast_sessions_loaded")
    decoder = _session(model_dir / codec_name, args.threads)
    record("slow_fast_codec_decoder_loaded")
    # Touch metadata so lazy session setup is included in the loaded measurement.
    _ = (slow.get_inputs(), fast.get_inputs(), decoder.get_inputs())
    record("all_session_metadata_touched")
    result = {
        "precision": args.precision,
        "codec_precision": args.codec_precision,
        "threads": args.threads,
        "model_dir": str(model_dir),
        "stages": stages,
        "loaded_rss_mib": stages[-1]["rss_mib"],
        "notes": (
            "RSS after loading sessions; no prompt, KV cache, generation, or codec decode. "
            "Compare this stage with the official 'after loading' claim."
        ),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    # Keep references alive until the final sample; ORT may otherwise release sessions.
    _ = (slow, fast, decoder, time.time())


if __name__ == "__main__":
    main()
