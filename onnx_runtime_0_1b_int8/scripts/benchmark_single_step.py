"""Measure single ONNX graph calls using the official model-card convention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arktts_runtime.runtime import ArkTtsRuntime


def stats(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "median_ms": statistics.median(values) * 1000,
        "p95_ms": sorted(values)[min(len(values) - 1, round((len(values) - 1) * 0.95))] * 1000,
        "mean_ms": statistics.mean(values) * 1000,
    }


def measure(args: argparse.Namespace) -> dict[str, object]:
    runtime = ArkTtsRuntime(args.model_dir, args.voices_dir, precision=args.precision, threads=args.threads)
    reference_codes, meta = runtime.voices.load(args.voice)
    prompt = runtime.prompt_builder.build(args.text, meta["reference_text"], reference_codes)

    def prefill() -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[float]]:
        state = runtime._empty_slow_state()
        timings: list[float] = []
        logits = hidden = None
        for position in range(prompt.shape[2]):
            started = time.perf_counter()
            logits, hidden = runtime._slow_step(prompt[:, :, position : position + 1], position, state)
            timings.append(time.perf_counter() - started)
        assert logits is not None and hidden is not None
        return state, logits, hidden, timings

    # Warm up graph/kernel caches and discard this run.
    state, logits, hidden, _ = prefill()
    warm_caches = runtime._empty_fast_caches()
    runtime._fast_step(hidden, 0, True, 0, warm_caches)
    for position in range(1, int(runtime.manifest["num_codebooks"])):
        runtime._fast_step(hidden, 0, False, position, warm_caches)

    prefill_timings: list[float] = []
    slow_decode_timings: list[float] = []
    fast_call_timings: list[float] = []
    fast_frame_timings: list[float] = []
    semantic_begin = int(runtime.manifest["semantic_begin_id"])
    for _ in range(args.repeats):
        state, logits, hidden, timings = prefill()
        prefill_timings.extend(timings)
        # Use greedy IDs. Sampling is intentionally excluded from graph timing.
        semantic = semantic_begin + int(np.argmax(np.asarray(logits).reshape(-1)[:-1]))
        caches = runtime._empty_fast_caches()
        frame_started = time.perf_counter()
        started = time.perf_counter()
        runtime._fast_step(hidden, 0, True, 0, caches)
        fast_call_timings.append(time.perf_counter() - started)
        token = 0
        for position in range(1, int(runtime.manifest["num_codebooks"])):
            started = time.perf_counter()
            values = runtime._fast_step(hidden, token, False, position, caches)
            fast_call_timings.append(time.perf_counter() - started)
            token = int(np.argmax(np.asarray(values).reshape(-1)))
        fast_frame_timings.append(time.perf_counter() - frame_started)
        column = np.concatenate([[semantic], np.zeros(int(runtime.manifest["num_codebooks"]), dtype=np.int64)]).reshape(1, -1, 1)
        started = time.perf_counter()
        runtime._slow_step(column, int(prompt.shape[2]), state)
        slow_decode_timings.append(time.perf_counter() - started)

    return {
        "backend": f"onnx-{args.precision}",
        "threads": args.threads,
        "text": args.text,
        "prompt_tokens": int(prompt.shape[2]),
        "fast_calls_per_frame": int(runtime.manifest["num_codebooks"]),
        "repeats": args.repeats,
        "prompt_prefill_slow_step": stats(prefill_timings),
        "generated_slow_step": stats(slow_decode_timings),
        "generated_fast_graph_call": stats(fast_call_timings),
        "generated_fast_frame_10_calls": stats(fast_frame_timings),
        "official_reference": {
            "threads": 8,
            "slow_ar_token_ms": 19,
            "fast_ar_frame_ms": 8,
            "prompt_prefill_token_ms": 19,
            "source": "Audio8/audio8-TTS-0.1B-ONNX-INT8 README.md",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=["int8", "u8u8", "w8a16"], required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--text", default="这是语言合成量化效果测试")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    result = measure(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
