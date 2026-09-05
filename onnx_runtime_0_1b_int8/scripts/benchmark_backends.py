"""Benchmark the 0.1B ONNX runtime or original PyTorch CPU model.

Run each backend in a separate process so peak RSS and load time are isolated.
The benchmark uses the same precomputed reference codes for both backends and
reports AR generation separately from codec decoding.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time

import numpy as np
import psutil


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


class PeakRss:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self.stop.wait(0.01):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self) -> "PeakRss":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def summarize(
    backend: str,
    load_seconds: float,
    runs: list[dict[str, float | int]],
    peak_rss: int,
    threads: int,
) -> dict[str, object]:
    generation = [float(run["generation_seconds"]) for run in runs]
    decoding = [float(run["decode_seconds"]) for run in runs]
    end_to_end = [float(run["end_to_end_seconds"]) for run in runs]
    rtfs = [float(run["rtf"]) for run in runs]
    frames_per_second = [float(run["frames_per_second"]) for run in runs]
    return {
        "backend": backend,
        "threads": threads,
        "load_seconds": load_seconds,
        "peak_rss_mib": peak_rss / 1024**2,
        "runs": runs,
        "median": {
            "generation_seconds": statistics.median(generation),
            "decode_seconds": statistics.median(decoding),
            "end_to_end_seconds": statistics.median(end_to_end),
            "rtf": statistics.median(rtfs),
            "frames_per_second": statistics.median(frames_per_second),
        },
        "p95": {
            "end_to_end_seconds": percentile(end_to_end, 0.95),
            "rtf": percentile(rtfs, 0.95),
        },
    }


def benchmark_onnx(args: argparse.Namespace) -> tuple[float, list[dict[str, float | int]]]:
    from arktts_runtime.runtime import ArkTtsRuntime

    started = time.perf_counter()
    runtime = ArkTtsRuntime(
        args.model_dir,
        args.voices_dir,
        precision=args.precision,
        threads=args.threads,
    )
    load_seconds = time.perf_counter() - started
    # Warm both AR graphs and the codec without including this run in results.
    warm_frames = list(
        runtime.iter_codes(
            text=args.text,
            voice=args.voice,
            max_new_tokens=min(4, args.max_new_tokens),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            seed=args.seed,
        )
    )
    runtime.decode_codes(np.stack(warm_frames, axis=1))

    runs: list[dict[str, float | int]] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        frames = list(
            runtime.iter_codes(
                text=args.text,
                voice=args.voice,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                seed=args.seed,
            )
        )
        generation_seconds = time.perf_counter() - started
        codes = np.stack(frames, axis=1)
        started_decode = time.perf_counter()
        audio = runtime.decode_codes(codes)
        decode_seconds = time.perf_counter() - started_decode
        end_to_end = generation_seconds + decode_seconds
        audio_seconds = len(audio) / int(runtime.manifest["sample_rate"])
        runs.append(
            {
                "frames": codes.shape[1],
                "audio_seconds": audio_seconds,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "end_to_end_seconds": end_to_end,
                "frames_per_second": codes.shape[1] / generation_seconds,
                "rtf": end_to_end / audio_seconds,
            }
        )
    return load_seconds, runs


def benchmark_pytorch(args: argparse.Namespace) -> tuple[float, list[dict[str, float | int]]]:
    import torch
    from transformers import AutoModel, AutoProcessor

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.pytorch_model, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.pytorch_model,
        trust_remote_code=True,
        dtype=torch.float32,
    ).eval().to("cpu")
    # ONNX construction loads its codec session eagerly. Do the equivalent
    # here so load_seconds ends when both backends are ready to synthesize.
    model.load_codec(device="cpu")
    load_seconds = time.perf_counter() - started

    voice_dir = args.voices_dir / args.voice
    meta = json.loads((voice_dir / "meta.json").read_text(encoding="utf-8"))
    reference_codes = np.load(voice_dir / "codes.npy", allow_pickle=False)
    inputs = processor(
        text=args.text,
        reference_text=meta["reference_text"],
        reference_codes=reference_codes,
        return_tensors="pt",
    )

    def generate(limit: int):
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        return model.generate(
            **inputs,
            max_new_tokens=limit,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            generator=generator,
            return_dict_in_generate=True,
        )

    warm = generate(min(2, args.max_new_tokens))
    model.decode_audio(warm.codes)
    runs: list[dict[str, float | int]] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        output = generate(args.max_new_tokens)
        generation_seconds = time.perf_counter() - started
        frames = int(output.code_lengths[0].item())
        started_decode = time.perf_counter()
        waveforms, lengths = model.decode_audio(output.codes)
        decode_seconds = time.perf_counter() - started_decode
        del waveforms
        audio_seconds = int(lengths[0].item()) / int(model.config.codec_sample_rate)
        end_to_end = generation_seconds + decode_seconds
        runs.append(
            {
                "frames": frames,
                "audio_seconds": audio_seconds,
                "generation_seconds": generation_seconds,
                "decode_seconds": decode_seconds,
                "end_to_end_seconds": end_to_end,
                "frames_per_second": frames / generation_seconds,
                "rtf": end_to_end / audio_seconds,
            }
        )
    return load_seconds, runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["onnx", "pytorch"], required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--pytorch-model", type=Path)
    parser.add_argument("--voices-dir", type=Path, required=True)
    parser.add_argument("--voice", default="default")
    parser.add_argument("--precision", choices=["int8", "u8u8", "w8a16"])
    parser.add_argument("--text", default="这是语言合成量化效果测试")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.threads < 1 or args.max_new_tokens < 1:
        parser.error("repeats, threads, and max-new-tokens must be positive")
    if args.backend == "onnx" and (args.model_dir is None or args.precision is None):
        parser.error("ONNX requires --model-dir and --precision")
    if args.backend == "pytorch" and args.pytorch_model is None:
        parser.error("PyTorch requires --pytorch-model")

    with PeakRss() as memory:
        if args.backend == "onnx":
            load_seconds, runs = benchmark_onnx(args)
            name = f"onnx-{args.precision}"
        else:
            load_seconds, runs = benchmark_pytorch(args)
            name = "pytorch-fp32"
    result = summarize(name, load_seconds, runs, memory.peak, args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
