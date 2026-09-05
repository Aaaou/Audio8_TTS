"""Dump pre-sampling ONNX fingerprints for cross-platform comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arktts_runtime.runtime import ArkTtsRuntime, _sample


def describe(value: np.ndarray) -> dict[str, object]:
    value = np.asarray(value)
    result: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "mean": float(np.mean(value)),
        "std": float(np.std(value)),
    }
    if np.issubdtype(value.dtype, np.floating):
        result["finite"] = bool(np.isfinite(value).all())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--precision", choices=["int8", "u8u8", "w8a16"], default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runtime = ArkTtsRuntime(
        args.model_dir, args.voices_dir, precision=args.precision, threads=args.threads
    )
    reference_codes, meta = runtime.voices.load(args.voice)
    prompt = runtime.prompt_builder.build(args.text, meta["reference_text"], reference_codes)
    state = runtime._empty_slow_state()
    logits = hidden = None
    prefill_logits: list[np.ndarray] = []
    prefill_hidden: list[np.ndarray] = []
    for position in range(prompt.shape[2]):
        logits, hidden = runtime._slow_step(prompt[:, :, position : position + 1], position, state)
        prefill_logits.append(np.asarray(logits).copy())
        prefill_hidden.append(np.asarray(hidden).copy())
    assert logits is not None and hidden is not None

    rng = np.random.default_rng(42)
    semantic = runtime._sample_semantic(logits, [], 0.3, 0.9, 50, rng)
    fast_caches = runtime._empty_fast_caches()
    fast_logits = [runtime._fast_step(hidden, 0, True, 0, fast_caches)]
    token = semantic - int(runtime.manifest["semantic_begin_id"])
    codes = [token]
    for position in range(1, int(runtime.manifest["num_codebooks"])):
        value = runtime._fast_step(hidden, token, False, position, fast_caches)
        fast_logits.append(value)
        token = _sample(value, 0.3, 0.9, 50, rng)
        codes.append(token)

    arrays: dict[str, np.ndarray] = {
        "prompt": prompt,
        "reference_codes": reference_codes,
        "slow_logits": logits,
        "slow_hidden": hidden,
        "prefill_logits": np.stack(prefill_logits),
        "prefill_hidden": np.stack(prefill_hidden),
        "semantic": np.asarray([semantic], dtype=np.int64),
        "codes": np.asarray(codes, dtype=np.int64),
    }
    arrays.update({f"fast_logits_{i}": value for i, value in enumerate(fast_logits)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), **arrays)
    summary = {
        "ort_version": __import__("onnxruntime").__version__,
        "threads": args.threads,
        "precision": runtime.precision,
        "prompt": describe(prompt),
        "reference_codes": describe(reference_codes),
        "slow_logits": describe(logits),
        "slow_hidden": describe(hidden),
        "semantic": semantic,
        "codes": codes,
        "fast_logits": [describe(value) for value in fast_logits],
    }
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
